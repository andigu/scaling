"""Base classes for quantum error correction codes.

This module provides the foundational abstract base class for implementing
quantum error correction codes, including the Code ABC and related utilities.
"""

import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Self, Tuple, Union, Iterable

import numpy as np
import pandas as pd
import stim

from ..types import Pauli

from ..instruction import Instruction
from ..measurement_tracker import MeasurementTracker, SyndromeCoordinate, SyndromeType
from ..utils import groupby
from .types import CodeMetadata, SyndromeExtractionProcedure


class Code(ABC):
    """Abstract base class for quantum error correction codes.

    This class provides the foundational interface for all quantum error
    correction codes, including methods for applying gates, measuring
    stabilizers, and managing qubit resources.
    """
    NAME = 'Code'
    TRANSVERSAL_1Q: List[str]
    TRANSVERSAL_2Q: List[str]
    
    def __init__(self, qubit_ids: OrderedDict[str, np.ndarray],
                 metadata: CodeMetadata, block_id: int = 0,
                 **code_params):
        """Initialize a Code with qubit IDs and metadata.

        Args:
            qubit_ids: Ordered dictionary mapping group names to qubit arrays
            metadata: Dictionary containing code-specific metadata
            block_id: Unique identifier for this logical qubit
            **code_params: Additional code-specific parameters
        """
        super().__init__()
        self.qubit_ids = qubit_ids
        self.metadata = metadata
        self.code_params = code_params
        self.block_id = block_id
    
    @staticmethod
    def allowable_code_params() -> Iterable[Dict[str, Any]]:
        return [{}]

    @classmethod
    def from_code_params(cls, block_id: int = 0, physical_id_start: int = 0,
                        **code_params) -> Self:
        """Create a Code instance from parameters.

        Args:
            block_id: Unique identifier for this logical qubit
            physical_id_start: Starting index for physical qubit allocation
            **code_params: Code-specific parameters

        Returns:
            Configured Code instance
        """
        qubit_ids = cls.get_qubit_ids(**code_params, index_start=physical_id_start)
        metadata = cls.create_metadata(**code_params)
        return cls(qubit_ids, metadata, block_id=block_id,
                  **code_params)
    
    @classmethod
    @lru_cache
    def _check_graph_edge_types(cls, allowable_code_params: Optional[Tuple[Tuple[str, Any]]] = None):
        all_edge_types = []
        if allowable_code_params is not None:
            cps = [{k: v for k, v in x} for x in allowable_code_params]
        else: cps = cls.allowable_code_params()
        for code_params in cps:
            tanner = cls.create_metadata(**code_params).tanner
            tanner = pd.DataFrame(tanner)
            check2 = pd.merge(tanner, tanner, on='data_id')
            check2 = check2[check2['check_id_x'] != check2['check_id_y']]
            
            for _, grp in check2.groupby(['check_id_x', 'check_id_y']):
                grp = grp.sort_values(by=['edge_type_x', 'edge_type_y'])
                all_edge_types.append(frozenset(zip(grp['edge_type_x'].tolist(), grp['edge_type_y'].tolist())))
        all_edge_types = list(set(all_edge_types))
        ret = {edge_type: i for i, edge_type in enumerate(all_edge_types)}
        # ret[tuple()] = 0 # Self-loop
        return ret
    
    @classmethod
    def check_graph_edge_types(cls, allowable_code_params: Optional[Iterable[Dict[str, Any]]] = None):
        # Convenience wrapper for _check_graph_edge_types to make allowable_code_params hashable for the cache
        if allowable_code_params is not None:
            cps = tuple([tuple(x.items()) for x in allowable_code_params])
        else:
            cps = None
        return cls._check_graph_edge_types(cps)

    @classmethod
    def check_graph(cls, edge_types: Optional[Dict[Any, int]] = None, **code_params) -> pd.DataFrame:
        tanner = pd.DataFrame(cls.create_metadata(**code_params).tanner)
        check2 = pd.merge(tanner, tanner, on='data_id')
        check2 = check2[check2['check_id_x'] != check2['check_id_y']]
        ret = []
        if edge_types is None: edge_types = cls.check_graph_edge_types([code_params])
        for (idx1, idx2), grp in check2.groupby(['check_id_x', 'check_id_y']):
            edge_type = frozenset(zip(grp['edge_type_x'].tolist(), grp['edge_type_y'].tolist()))
            ret.append((idx1, idx2, edge_types[edge_type]))
        # ret.extend([(idx, idx, 0) for idx in tanner['check_id'].unique()]) # Add self-loops
        return pd.DataFrame(ret, columns=['check_id_x', 'check_id_y', 'edge_type'])

    @staticmethod
    @abstractmethod
    def create_metadata(**_) -> CodeMetadata: ...

    @property
    def all_qubits(self):
        """Get all physical qubits used by this code."""
        return np.concatenate([self.qubit_ids[k] for k in self.qubit_ids.keys()])

    def copy(self):
        """Create a deep copy of this Code instance."""
        return type(self)(
            qubit_ids=OrderedDict({k: v.copy() for k, v in self.qubit_ids.items()}),
            metadata=self.metadata.copy(),
            block_id=self.block_id,
            **copy.deepcopy(self.code_params)
        )

    def __str__(self):
        params_str = f'id={self.block_id}, params={self.code_params}'
        return f'<{self.NAME} {params_str}>'

    def __repr__(self):
        return self.__str__()

    @classmethod
    @lru_cache
    def _initialize(cls, paulis: Tuple[int, ...], signs: Tuple[bool, ...], **code_params):
        # targets in the returned instructions are indices into the data qubit id's
        # paulis is a list of 'X' or 'Z', whose length is equal to the number of logical qubits
        # in the code block for cls
        metadata = cls.create_metadata(**code_params)
        check_ids, info = groupby(metadata.tanner, by='check_id')
        stabs = []
        for group in info:
            stabs.append(stim.PauliString('*'.join([f'{str(Pauli(p))}{i}' for (p, i) in zip(group['pauli'], group['data_id'])])))
        for i in range(len(paulis)):
            obs = metadata.logical_operators[
                (metadata.logical_operators['logical_id'] == i) &
                (metadata.logical_operators['logical_pauli'] == paulis[i])
            ]
            sign_char = "+" if signs[i] else "-"
            stabs.append(stim.PauliString(sign_char + '*'.join([f'{str(Pauli(p))}{i}' for (p, i) in zip(obs['physical_pauli'], obs['data_id'])])))
        ret = [Instruction('R', np.arange(metadata.num_data))]
        circ = stim.Tableau.from_stabilizers(stabs, allow_redundant=True).to_circuit()
        for instr in circ:
            name = instr.name
            if isinstance(instr, stim.CircuitInstruction):
                targets = np.array([x.qubit_value for x in instr.targets_copy()])
                if len(instr.target_groups()[0]) == 2:  # Two-qubit gate
                    targets = targets.reshape((-1, 2))
                ret.append(Instruction(name, targets, meta={'noiseless': True}))
            else:
                raise ValueError(f"Unexpected instruction type: {type(instr)}")
        return ret
    
    @property
    def num_logical_qubits(self):
        return self.metadata.num_logical_qubits

    def initialize(self, basis: Union[int, Iterable[int], None]=None, signs: Union[List[bool], bool, None]=None, 
                   qubit_group: str = 'data', reset_others: bool = True) -> List[Instruction]:
        basis_list = []
        if basis is None:
            basis_list = [Pauli.Z] * self.num_logical_qubits
        if signs is None:
            signs = [True] * self.num_logical_qubits
        if isinstance(basis, (int, np.integer)):
            basis_list = [basis] * self.num_logical_qubits
        if isinstance(signs, bool):
            signs = [signs] * len(basis_list)

        ret = [
            Instruction(instr.name, self.qubit_ids[qubit_group][instr.targets], meta={'noiseless': True})
                for instr in self._initialize(tuple(basis_list), tuple(signs), **self.code_params)
        ]
        if reset_others:
            if any(k != qubit_group for k in self.qubit_ids.keys()):
                outside_group = np.concatenate([v for k, v in self.qubit_ids.items() if k != qubit_group])
            ret += [
                Instruction('R', outside_group, meta={'noiseless': True})
            ]
        return ret

    def measure_stabilizers(self,
                           meas_track: Optional[MeasurementTracker] = None,
                           **kwargs) -> List[Instruction]:
        """Measure the stabilizers of this code.

        Args:
            meas_track: Optional measurement tracker for detector construction
            **kwargs: Additional arguments (e.g., time)

        Returns:
            List of Instructions for stabilizer measurements
        """
        time = kwargs.get('time', -1)
        all_qubits = self.all_qubits
        params = self.code_params
        se_data = type(self).get_syndrome_extraction_instructions(**params)
        measurement_ids = []

        for subcycle, (check_ids, syndrome_type) in enumerate(zip(se_data.syndrome_ids, se_data.syndrome_types)):
            if meas_track is not None:
                measurement_ids.append(meas_track.register_measurements(
                    time=time,
                    block_id=self.block_id,
                    syndrome_type=syndrome_type,
                    syndrome_ids=check_ids,
                    data_ids=se_data.data_ids[subcycle] if se_data.data_ids is not None else None,
                    subcycle=subcycle
                ))
            else:
                measurement_ids.append(None)

        measurement_ids = iter(measurement_ids)
        ret = []
        for instr in se_data.instructions:
            if instr.name in Instruction.MEASURE:
                meas_meta = {'measurement_ids': next(measurement_ids)}
                meas_meta.update(**{k: v for k, v in instr.meta.items() if k != 'measurement_ids'})
                ret.append(Instruction(instr.name, all_qubits[instr.targets],
                                     meta=meas_meta))
            else:
                ret.append(instr.copy(target_remapper=all_qubits))
        return ret

    def apply_gate(self, gate: str, *args, **kwargs) -> List[Instruction]:
        """Apply a logical gate to this code.

        Args:
            gate: Name of the gate to apply (e.g., 'H', 'X', 'CX')
            *args: Additional arguments for the gate
            **kwargs: Additional keyword arguments for the gate

        Returns:
            List of Instructions implementing the logical gate

        Raises:
            ValueError: If the code does not support the specified gate
        """
        method_name = f'_apply_{gate.lower()}'
        if hasattr(self, method_name):
            return getattr(self, method_name)(*args, **kwargs)
        else:
            raise ValueError(f'Code {self.NAME} does not support gate {gate}')

    def _apply_swap(self, target: Self, **_) -> List[Instruction]:
        """Apply a SWAP gate between this code and target code."""
        swap_targets = np.stack([self.qubit_ids['data'],
                               target.qubit_ids['data']], axis=1)
        return [Instruction('SWAP', swap_targets)]

    def _get_flags(self, meas_track: MeasurementTracker, time: int):
        """Add additional detectors at the end of each round (e.g., flags)

        Override in subclasses to add custom detector patterns.
        """
        if 'flag' in self.metadata.entities:
            flag_ids = self.metadata.entities['flag']['flag_id']
            patterns = [SyndromeCoordinate(time=time, block_id=self.block_id, syndrome_type=SyndromeType.FLAG, ids=flag_ids)]
            meas_track.register_detectors(patterns)

    def _swap_detectors(self, meas_track: MeasurementTracker, target: Self,
                       time: int):
        """Register detectors for SWAP gate between codes."""
        ctrl, targ = self.block_id, target.block_id
        check_ids = self.metadata.check['check_id']
        patterns = [
            [SyndromeCoordinate(time=time-1, block_id=ctrl, syndrome_type=SyndromeType.CHECK, ids=check_ids),
             SyndromeCoordinate(time=time, block_id=targ, syndrome_type=SyndromeType.CHECK, ids=check_ids)],
            [SyndromeCoordinate(time=time-1, block_id=targ, syndrome_type=SyndromeType.CHECK, ids=check_ids),
             SyndromeCoordinate(time=time, block_id=ctrl, syndrome_type=SyndromeType.CHECK, ids=check_ids)]
        ]
        for pattern in patterns:
            meas_track.register_detectors(pattern)

    def _apply_i(self, **_) -> List[Instruction]:
        """Apply identity gate to data qubits."""
        return [Instruction('I', self.qubit_ids['data'])]

    def _i_detectors(self, meas_track: MeasurementTracker, time: int):
        """Register detectors for identity gate (no-op)."""
        check_ids = self.metadata.check['check_id']
        patterns = [
            [SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=check_ids),
            SyndromeCoordinate(time=time, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=check_ids)]
        ]
        for pattern in patterns:
            meas_track.register_detectors(pattern)

    @property
    def num_qubits(self):
        """Return the total number of physical qubits used by this code."""
        return sum(len(v) for v in self.qubit_ids.values())

    @staticmethod
    @abstractmethod
    def get_qubit_ids(index_start: int = 0,
                     **code_params) -> OrderedDict[str, np.ndarray]:
        """Get qubit IDs for this code.

        Ordered dictionary maintains ID generation order for consistent
        concatenation and memoization. Must at *least* contain 'data'
        """
        pass

    @staticmethod
    @abstractmethod
    def get_syndrome_extraction_instructions(**code_params) -> \
            SyndromeExtractionProcedure:
        """Get syndrome extraction instructions for this code.

        Returns a SyndromeExtractionData object containing:
        - instructions: List of Instructions for syndrome extraction
        - targets: List of target qubit arrays for each measurement group
        - types: List of qubit types being measured
        - groups: List of qubit group names
        - roles: List of measurement roles (from MeasurementRole enum)

        Args:
            **code_params: Parameters specific to the code implementation

        Returns:
            SyndromeExtractionData: Structured data for syndrome extraction
        """
        pass

    @abstractmethod
    def measure_data(self, meas_track: Optional[MeasurementTracker] = None,
                    basis: str = 'Z', **kwargs) -> List[Instruction]:
        """Measure the data qubits in the given basis.

        Returns:
            Instructions for data measurements
        """
        pass

    def construct_detectors(self, meas_track: MeasurementTracker,
                           gate: str, time: int, **kwargs):
        """Construct detectors for the code."""
        method_name = f'_{gate.lower()}_detectors'
        if hasattr(self, method_name):
            detector_method = getattr(self, method_name)
            detector_method(meas_track=meas_track, time=time, **kwargs)
            self._get_flags(meas_track=meas_track, time=time)
            if 'target' in kwargs:
                target: Code = kwargs['target']
                # Call protected method on target - this is intentional for QEC
                target._get_flags(meas_track=meas_track, time=time)
        else:
            error_msg = f'Code {self.NAME} does not support gate {gate}'
            raise ValueError(error_msg)
