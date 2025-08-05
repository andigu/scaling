"""Surface code implementations for quantum error correction.

This module implements various surface code variants including the standard
surface code and Steane surface code. Surface codes are topological quantum
error correction codes defined on 2D lattices.
"""

from collections import OrderedDict
from functools import lru_cache
from typing import List, Iterable, Dict, Any
import numpy as np
import pandas as pd

from ..types import Pauli

from .css import CSSCode
from .decorators import supports_transversal_gates
from .steane_mixin import SteaneMixin
from .types import SyndromeExtractionProcedure, CodeMetadata
from ..instruction import Instruction
from ..measurement_tracker import MeasurementTracker, SyndromeCoordinate, SyndromeType

@supports_transversal_gates(gates_1q=['I', 'X', 'Z', 'H'], gates_2q=['CX', 'SWAP'])
class SurfaceCodeBase(CSSCode):
    """Abstract base class for surface code implementations.
    
    Provides common functionality for surface code variants including geometric
    layout generation, lattice visualization, and transversal Hadamard gate
    implementation. Surface codes are CSS codes defined on 2D square lattices
    with data qubits on vertices and check qubits on faces/edges.
    
    Concrete subclasses must implement syndrome extraction and initialization
    procedures specific to their measurement protocols (e.g., standard vs
    Steane-style syndrome extraction).
    """
    NAME = 'SurfaceCode'
    
    @staticmethod
    def allowable_code_params() -> Iterable[Dict[str, Any]]:
        return [{'d': d} for d in range(3, 33, 2)]

    @staticmethod
    @lru_cache
    def create_metadata(d, **_) -> CodeMetadata:
        data_indices = np.arange(d**2)
        ancilla_indices = np.arange(d**2-1)
        data_metadata, obs_metadata = [], []
        for row in range(d):
            for col in range(d):
                if col == d//2: # *IMPORTANT*: USE CENTERED LOGICAL OBSERVABLES
                    obs_metadata.append((0, Pauli.Z, data_indices[row * d + col], Pauli.Z))
                if row == d//2:
                    obs_metadata.append((0, Pauli.X, data_indices[row * d + col], Pauli.X))
                data_metadata.append({
                    'data_id': data_indices[row * d + col],
                    'pos_x': col,
                    'pos_y': row,
                })
        obs_metadata = pd.DataFrame(obs_metadata, columns=['logical_id', 'logical_pauli', 'data_id', 'physical_pauli'])
        # Create ancilla grid layout
        ancilla_grid = np.full((d + 1, d + 1), -1)

        # Place ancilla qubits in the grid
        ancilla_count = 0
        # Top boundary
        for col in range(1, d, 2):
            ancilla_grid[0, col] = ancilla_indices[ancilla_count]
            ancilla_count += 1
        # Middle rows
        for row in range(1, d):
            ancilla_grid[row, (row%2):d+(row%2)] = ancilla_indices[ancilla_count:ancilla_count+d]
            ancilla_count += d
        # Bottom boundary
        for col in range(2, d + 1, 2):
            ancilla_grid[d, col] = ancilla_indices[ancilla_count]
            ancilla_count += 1
        # Generate metadata for each ancilla
        local_face_id = 0
        tanner = []
        check = []
        for row in range(d + 1):
            for col in range(d + 1):
                if ancilla_grid[row, col] != -1:
                    check_type = Pauli.X if (row + col) % 2 == 0 else Pauli.Z
                    check.append({
                        'check_id': local_face_id,
                        'pos_x': col,
                        'pos_y': row,
                        'check_type': check_type,
                    })
                    # Top-left, top-right, bottom-left, bottom-right
                    dz = [(-1, -1), (-1, 0), (0, -1), (0, 0)]
                    for orientation_index, (dr, dc) in enumerate(dz):
                        if 0 <= row + dr < d and 0 <= col + dc < d:
                            tanner.append({
                                'check_id': local_face_id,
                                'data_id': data_indices[(row + dr) * d + (col + dc)],
                                'edge_type': orientation_index,
                                'pauli': check_type,
                            })
                    local_face_id += 1

        return CodeMetadata(
            tanner=pd.DataFrame(tanner).to_records(index=False),
            logical_operators=obs_metadata.to_records(index=False),
            data=pd.DataFrame(data_metadata).to_records(index=False),
            check=pd.DataFrame(check).to_records(index=False)
        )

    def _apply_h(self, **_) -> List[Instruction]:
        """
        Applies a logical Hadamard gate to the surface code.

        This involves applying physical Hadamard gates to data qubits and
        then performing a permutation (SWAP network) to effectively rotate
        the code.

        Returns:
            List of `Instruction` objects representing the physical operations.
        """
        d = self.code_params['d']
        data_ids = self.qubit_ids['data']
        new_data = np.rot90(data_ids.reshape((d, d)), k=1).flatten()
        perm = {data_ids[i]: new_data[i] for i in range(len(data_ids))}
        return [Instruction('H', data_ids),
                Instruction('SWAP', _get_transpositions(perm), meta={'noiseless': True})]

    def _h_detectors(self, meas_track: MeasurementTracker, time: int):
        """
        Registers detectors for a logical Hadamard gate.

        This method defines the detector patterns that relate measurements
        before and after a logical Hadamard gate. For surface codes, this
        involves correlating X-type and Z-type stabilizers across time,
        reflecting the basis change.

        Args:
            meas_track: The `MeasurementTracker` instance to register detectors with.
            time: The current time step of the simulation.
        """
        d, check = self.code_params['d'], self.metadata.check
        check_id_grid = np.full((d + 1, d + 1), -1)
        check_id_grid[check['pos_x'], check['pos_y']] = check['check_id']
        pairings = np.stack([np.rot90(check_id_grid, k=3), check_id_grid], axis=-1)
        pairings = pairings[check['pos_x'], check['pos_y']]
        detector_units = [SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=pairings[...,0]),
             SyndromeCoordinate(time=time, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=pairings[...,1])]
        meas_track.register_detectors(detector_units)
    

class SurfaceCode(SurfaceCodeBase):
    """Standard surface code implementation with dedicated ancilla qubits.
    
    Uses separate ancilla qubits for syndrome extraction via a 4-step CNOT
    schedule that measures stabilizers non-destructively. This is the most
    common surface code variant, requiring d² data qubits and d²-1 ancilla
    qubits for distance d.
    
    The code follows the geometry convention from:
    https://content.cld.iop.org/journals/1367-2630/20/4/043038/revision2/njpaab341f1_hr.jpg
    """
    @staticmethod
    def get_qubit_ids(index_start: int = 0, **code_params) -> OrderedDict[str, np.ndarray]:
        d = code_params['d']
        data_indices = np.arange(index_start, index_start + d ** 2)
        ancilla_indices = np.arange(index_start + d ** 2, index_start + 2 * d ** 2 - 1)
        return OrderedDict([('data', data_indices), ('check_ancilla', ancilla_indices)])

    @lru_cache
    @staticmethod
    def get_syndrome_extraction_instructions(d: int) -> SyndromeExtractionProcedure:
        """
        Returns the syndrome extraction instructions for the standard surface code.

        This method generates a sequence of Stim-compatible instructions for
        measuring the stabilizers of a surface code of given distance `d`.
        It typically involves ancilla initialization, CNOT schedules, and final
        ancilla measurements.

        Args:
            d: The distance of the surface code.

        Returns:
            A `SyndromeExtractionProcedure` object containing the instructions
            and metadata for syndrome extraction.
        """
        ret = []
        data_indices, ancilla_indices = np.arange(d**2), np.arange(d**2, 2*d**2-1)
        metadata = SurfaceCode.create_metadata(d)
        tanner = metadata.tanner
        ret.append(Instruction('R', ancilla_indices))
        x_ancilla = ancilla_indices[np.unique(tanner[tanner['pauli'] == Pauli.X]['check_id'])]
        ret.append(Instruction('H', x_ancilla))
        ordering = {
            Pauli.X: np.array([2, 0, 3, 1]),
            Pauli.Z: np.array([2, 3, 0, 1])
        }
        ordering = np.array([ordering[pauli] for pauli in tanner['pauli']])
        for stabilizer_step in range(4):
            matching_rows = tanner[tanner['edge_type'] == ordering[:, stabilizer_step]]
            ancilla_qubits = ancilla_indices[matching_rows['check_id']] # Implicitly sets ancilla_indices[i] <-> check_id=i
            data_qubits = data_indices[matching_rows['data_id']]
            measurement_types = matching_rows['pauli']
            control_qubits = np.where(measurement_types == Pauli.Z, data_qubits, ancilla_qubits)
            target_qubits = np.where(measurement_types == Pauli.Z, ancilla_qubits, data_qubits)
            ret.append(Instruction('CX', np.stack([control_qubits, target_qubits], axis=1)))

        # Convert X-basis measurements
        ret.append(Instruction('H', x_ancilla))
        ret.append(Instruction('M', ancilla_indices))
        return SyndromeExtractionProcedure(
            instructions=ret,
            syndrome_ids=[metadata.check['check_id'].tolist()],
            syndrome_types=[SyndromeType.CHECK],
        )

class SteaneSurfaceCode(SteaneMixin, SurfaceCodeBase):
    """Surface code variant using Steane-style syndrome extraction.
    
    Differs from standard surface codes by using teleportation-based
    syndrome extraction without dedicated ancilla qubits. Requires
    d² + d² qubits total, implementing fault-tolerant measurement
    through state preparation and Bell measurements.
    """
    NAME = 'SteaneSurfaceCode'


def _get_transpositions(perm: dict[int, int]) -> np.ndarray:
    """
    Decomposes a permutation into a sequence of transpositions (swaps).
    
    Args:
        perm: A dictionary representing the permutation, where keys are
              original indices and values are new indices.

    Returns:
        A NumPy array of shape (N, 2) where each row represents a
        transposition (pair of indices to swap).
    """
    # First find cycle decomposition of the permutation
    visited = set()
    cycles = []
    for start in perm:
        if start in visited:
            continue
        cycle = [start]
        visited.add(start)
        current = perm[start]
        while current != start:
            cycle.append(current)
            visited.add(current)
            current = perm[current]
        if len(cycle) > 1:  # Only include non-trivial cycles
            cycles.append(cycle)
    # Decompose each cycle into transpositions
    # A cycle (a1 a2 ... an) = (a1 an)(a1 an-1)...(a1 a2)
    transpositions = []
    for cycle in cycles:
        for i in range(len(cycle) - 1, 0, -1):
            transpositions.append((cycle[0], cycle[i]))

    return np.array(transpositions)
