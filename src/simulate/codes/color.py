"""Color code implementation for quantum error correction.

This module implements the color code, a topological quantum error correction
code defined on a honeycomb lattice. The implementation follows conventions
from recent literature and provides syndrome extraction and logical operations.
"""

from collections import OrderedDict
from collections.abc import Iterable
from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from ..types import Pauli

from .css import CSSCode
from .decorators import supports_transversal_gates
from .types import SyndromeExtractionProcedure, CodeMetadata
from ..instruction import Instruction
from ..measurement_tracker import MeasurementTracker, SyndromeCoordinate, SyndromeType

@supports_transversal_gates(
    gates_1q=['I', 'X', 'Y', 'Z', 'H', 'S', 'S_DAG'],
    gates_2q=['CX', 'SWAP']
)
class ColorCode(CSSCode):
    """Implementation of a Color Code for quantum error correction.

    Follows the conventions from:
    - https://arxiv.org/pdf/1911.00355
    - Honeycomb lattice geometry for color codes
    """
    NAME = 'ColorCode'

    @staticmethod
    def allowable_code_params() -> Iterable[Dict[str, Any]]:
        return [{'d': d} for d in range(3, 33, 2)]
    
    @staticmethod
    def get_qubit_ids(index_start: int = 0, **code_params) -> OrderedDict[str, np.ndarray]:
        """Get qubit IDs for color code of distance d."""
        d = code_params['d']
        coords = ColorCode.coordinates(d)
        data_coords = coords['data']
        flag_coords = coords['flag']
        check_coords = coords['face']

        ctr = index_start
        data_indices = np.arange(ctr, ctr + len(data_coords))
        ctr += len(data_coords)
        checks = np.arange(ctr, ctr + len(check_coords))
        ctr += len(check_coords)
        flags = np.arange(ctr, ctr + len(flag_coords))

        return OrderedDict([
            ('data', data_indices),
            ('check', checks),
            ('flag', flags)
        ])

    @lru_cache
    @staticmethod
    def create_metadata(d: int) -> CodeMetadata:
        tanner, logical_operators = [], []
        data, check = [], []
        entities = {
            'face': [],
            'flag': [],
        }
        relations = {
            'flag_data': [],
        }
        
        coords = ColorCode.coordinates(d)
        flag_coords, face_coords, data_coords = coords['flag'], coords['face'], coords['data']
        check_id = 0
        for pauli in [Pauli.X, Pauli.Z]: # Each face will have two stabilizers
            flag_id_offset = 0 if pauli == Pauli.X else len(flag_coords)
            for face_id, face_coord in enumerate(face_coords):
                # Flag support - find which flag qubits this check will interact with
                check_to_flag_distances = flag_coords - face_coord
                distances = np.linalg.norm(check_to_flag_distances, axis=-1)
                nearby_flag_indices = np.argwhere(np.isclose(
                    distances, 1 / np.sqrt(3))).flatten()
                for flag_id in nearby_flag_indices:
                    flag_to_data_distances = data_coords - flag_coords[flag_id]
                    nearby_data_indices = np.argwhere(np.isclose(np.linalg.norm(
                        flag_to_data_distances, axis=-1), 1/np.sqrt(3))).flatten()
                    for data_id in nearby_data_indices:
                        relative_position = None
                        if flag_to_data_distances[data_id, 1] > 0:
                            relative_position = 0
                        elif flag_to_data_distances[data_id, 0] > 0:
                            relative_position = 1
                        else:
                            relative_position = 2
                        relations['flag_data'].append({
                            'flag_id': flag_id + flag_id_offset, 
                            'data_id': data_id, 
                            'relative_position': relative_position
                        })
                    relative_position = None
                    if check_to_flag_distances[flag_id, 0] > 0:
                        relative_position = 0
                    elif check_to_flag_distances[flag_id, 1] < 0:
                        relative_position = 1
                    else:
                        relative_position = 2
                    flag_id_offset = 0 if pauli == Pauli.X else len(flag_coords)
                    entities['flag'].append({
                        'flag_id': flag_id + flag_id_offset, 
                        'pos_x': flag_coords[flag_id, 0],
                        'pos_y': flag_coords[flag_id, 1],
                        'face_id': face_id,
                        'flag_type': pauli,
                        'relative_position': relative_position,
                    })
                

                # Data support - find which data qubits belong to this stabilizer
                cd_distances = data_coords - face_coord
                distances = np.linalg.norm(cd_distances, axis=-1)
                nearby_data_indices = np.argwhere(np.isclose(distances, 1)).flatten()
                if pauli == Pauli.X: # Only add faces once
                    entities['face'].append({
                        'pos_x': face_coords[face_id, 0], 
                        'pos_y': face_coords[face_id, 1], 
                        'face_id': face_id
                    })
                for data_id in nearby_data_indices:
                    relative_position = None
                    if cd_distances[data_id, 0] > 0 and cd_distances[data_id, 1] > 0:
                        relative_position = 0
                    elif cd_distances[data_id, 0] > 0 and np.isclose(cd_distances[data_id, 1], 0):
                        relative_position = 1
                    elif cd_distances[data_id, 0] > 0 and cd_distances[data_id, 1] < 0:
                        relative_position = 2
                    elif cd_distances[data_id, 0] < 0 and cd_distances[data_id, 1] < 0:
                        relative_position = 3
                    elif cd_distances[data_id, 0] < 0 and np.isclose(cd_distances[data_id, 1], 0):
                        relative_position = 4
                    else:
                        relative_position = 5        
                    tanner.append({
                        'check_id': check_id, 
                        'data_id': data_id, 
                        'edge_type': relative_position,
                        'pauli': pauli
                    })
                check.append({
                    'check_id': check_id,
                    'face_id': face_id,
                    'check_type': pauli
                })
                check_id += 1
        
        for data_id in range(len(data_coords)):
            x, y = data_coords[data_id]
            data.append({
                'data_id': data_id,
                'pos_x': x,
                'pos_y': y,
            })
            if np.isclose(y, np.sqrt(3) * x):
                logical_operators.append([0, Pauli.Z, data_id, Pauli.Z])
            if np.isclose(y, -np.sqrt(3) * (x - np.max(data_coords[:, 0]))):
                logical_operators.append([0, Pauli.X, data_id, Pauli.X])
        logical_operators = pd.DataFrame(logical_operators, columns=['logical_id', 'logical_pauli', 'data_id', 'physical_pauli'])
        return CodeMetadata(
            tanner=pd.DataFrame(tanner).to_records(index=False),
            logical_operators=pd.DataFrame(logical_operators).to_records(index=False),
            data=pd.DataFrame(data).to_records(index=False),
            check=pd.DataFrame(check).to_records(index=False),
            entities={
                'face': pd.DataFrame(entities['face']).to_records(index=False),
                'flag': pd.DataFrame(entities['flag']).to_records(index=False),
            },
            relations={
                'flag_data': pd.DataFrame(relations['flag_data']).to_records(index=False),
            },
        )

    @lru_cache
    @staticmethod
    def coordinates(d: int) -> dict[str, np.ndarray]:
        """
        Generate the coordinates of data qubits in the Color Code lattice.

        Args:
            d: Code distance

        Returns:
            Array of [x,y] coordinates for data qubits
        """
        data_coords = []
        n_qubits_on_row = d
        for row_index in range(3 * (d - 1) // 2 + 1):
            if row_index % 3 == 0:
                data_coords.append([1.5 * (row_index // 3), row_index * np.sqrt(3) / 2])
                for qubit_in_row in range(1, n_qubits_on_row):
                    data_coords.append(
                        data_coords[-1] + np.array([2 if qubit_in_row % 2 == 1 else 1, 0]))
                n_qubits_on_row -= 1
            elif row_index % 3 == 1:
                data_coords.append([1.5 * (row_index // 3) + 0.5, row_index * np.sqrt(3) / 2])
                for qubit_in_row in range(1, n_qubits_on_row):
                    data_coords.append(
                        data_coords[-1] + np.array([1 if qubit_in_row % 2 == 1 else 2, 0]))
                n_qubits_on_row -= 1
            else:
                data_coords.append([1.5 * (row_index // 3) + 2, row_index * np.sqrt(3) / 2])
                for qubit_in_row in range(1, n_qubits_on_row):
                    data_coords.append(
                        data_coords[-1] + np.array([1 if qubit_in_row % 2 == 1 else 2, 0]))
        i = 0
        flag_coords = []
        for n_qubits_on_row in range(d + ((d - 1) // 2 - 1), 0, -1):
            flag_coords.append([(i + 1) / 2, i * np.sqrt(3) / 2 + 1 / (2 * np.sqrt(3))])
            i += 1
            for _ in range(1, n_qubits_on_row):
                flag_coords.append(flag_coords[-1] + np.array([1, 0]))
        face_coords = []
        n_qubits_on_row = (d - 1) // 2
        for i in range(3 * (d - 1) // 2):
            if i % 3 == 0:
                face_coords.append([1.5 * (i // 3) + 1, i * np.sqrt(3) / 2])
                for _ in range(1, n_qubits_on_row):
                    face_coords.append(face_coords[-1] + np.array([3, 0]))
            elif i % 3 == 1:
                face_coords.append(
                    [1.5 * (i // 3) + 3 / 2 + 1, i * np.sqrt(3) / 2])
                for _ in range(1, n_qubits_on_row):
                    face_coords.append(face_coords[-1] + np.array([3, 0]))
            else:
                face_coords.append([1.5 * (i // 3) + 1, i * np.sqrt(3) / 2])
                for _ in range(1, n_qubits_on_row):
                    face_coords.append(face_coords[-1] + np.array([3, 0]))
                n_qubits_on_row -= 1
        return {
            'data': np.array(data_coords),
            'flag': np.array(flag_coords),
            'face': np.array(face_coords),
        }

    def _h_detectors(self, meas_track: MeasurementTracker, time: int):
        """
        Registers detectors for a logical Hadamard gate.

        For color codes, this involves correlating X-type and Z-type stabilizers
        across time, reflecting the basis change.

        Args:
            meas_track: The `MeasurementTracker` instance.
            time: The current time step.
        """
        x_check_ids, z_check_ids = self.check_ids(Pauli.X), self.check_ids(Pauli.Z)
        patterns = [
            [SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=x_check_ids),
             SyndromeCoordinate(time=time, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=z_check_ids)],
            [SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=z_check_ids),
             SyndromeCoordinate(time=time, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=x_check_ids)]
        ]
        for pattern in patterns:
            meas_track.register_detectors(pattern)

    def _apply_h(self, **_) -> List[Instruction]:
        """
        Applies a logical Hadamard gate to the data qubits.

        Returns:
            A list containing a single `H` instruction on the data qubits.
        """
        return [Instruction('H', self.qubit_ids['data'])]

    def _s_detectors(self, meas_track: MeasurementTracker, time: int):
        """
        Registers detectors for a logical S gate.

        Args:
            meas_track: The `MeasurementTracker` instance.
            time: The current time step.
        """
        x_check_ids, z_check_ids = self.check_ids(Pauli.X), self.check_ids(Pauli.Z)
        patterns = [
            [SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=x_check_ids),
             SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=z_check_ids),
             SyndromeCoordinate(time=time, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=x_check_ids)],
            [SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=z_check_ids),
             SyndromeCoordinate(time=time, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=z_check_ids)]
        ]
        for pattern in patterns:
            meas_track.register_detectors(pattern)

    def _s_dag_detectors(self, meas_track: MeasurementTracker, time: int):
        """
        Registers detectors for a logical S_DAG gate.

        This method reuses the `_s_detectors` logic as S and S_DAG have
        the same detector patterns.

        Args:
            meas_track: The `MeasurementTracker` instance.
            time: The current time step.
        """
        self._s_detectors(meas_track, time)

    def _apply_s(self, **_) -> List[Instruction]:
        """
        Applies a logical S gate to the data qubits.

        Returns:
            A list containing a single `S_DAG` instruction on the data qubits.
        """
        return [Instruction('S_DAG', self.qubit_ids['data'])]

    def _apply_s_dag(self, **_) -> List[Instruction]:
        """
        Applies a logical S_DAG gate to the data qubits.

        Returns:
            A list containing a single `S` instruction on the data qubits.
        """
        return [Instruction('S', self.qubit_ids['data'])]

    def _y_detectors(self, meas_track: MeasurementTracker, time: int):
        """
        Registers detectors for a logical Y gate.

        This method defaults to `_i_detectors` as logical Y doesn't introduce
        additional specific detector patterns beyond identity.

        Args:
            meas_track: The `MeasurementTracker` instance.
            time: The current time step.
        """
        self._i_detectors(meas_track, time)

    def _apply_y(self, **_) -> List[Instruction]:
        """
        Applies a logical Y gate to the data qubits.

        Returns:
            A list containing a single `Y` instruction on the data qubits.
        """
        return [Instruction('Y', self.qubit_ids['data'])]

    @lru_cache
    @staticmethod
    def get_syndrome_extraction_instructions(d: int) -> SyndromeExtractionProcedure:
        """
        Returns the syndrome extraction instructions for the color code.

        This implementation is based on Figure 3 and 4 of
        https://arxiv.org/pdf/1911.00355. It involves a schedule of physical
        CX gates and measurements on check and flag qubits.

        Args:
            d: The distance of the color code.

        Returns:
            A `SyndromeExtractionProcedure` object containing the instructions
            and metadata for syndrome extraction.
        """
        metadata = ColorCode.create_metadata(d)
        qubit_ids = ColorCode.get_qubit_ids(d=d)
        ret = []

        check_ids = np.concatenate([qubit_ids['check'], qubit_ids['check']]) # Reuse qubits
        flag_ids = np.concatenate([qubit_ids['flag'], qubit_ids['flag']])
        data_ids = qubit_ids['data']
        # See Fig. 4 of https://arxiv.org/pdf/1911.00355
        schedule = [
            [None,        None,        ('cf', None)],
            [('cf', None), None,        ('fd', 0)  ],
            [('fd', 1),   ('cf', None), ('fd', 2)  ],
            [('fd', 0),   ('fd', 2),   ('cf', None)],
            [('cf', None), ('fd', 1),   None       ],
            [None,        ('cf', None), None       ]
        ]
        all_check_ids = []
        for basis in [Pauli.X, Pauli.Z]:
            flag_data = metadata.relations['flag_data']
            flag_face = metadata.entities['flag']
            flag_face = flag_face[flag_face['flag_type'] == basis]
            flag_data = pd.merge(pd.DataFrame(flag_data), pd.DataFrame(flag_face), on='flag_id', suffixes=('_data', '_face'), how='right')
            
            ret.append(Instruction('R' if basis == Pauli.Z else 'RX', check_ids))
            ret.append(Instruction('RX' if basis == Pauli.Z else 'R', flag_ids))
            for step in schedule:
                targets = []
                for orientation_check, x in enumerate(step):
                    if x is None:
                        continue
                    name, orientation_flag = x
                    if name == 'cf':
                        tmp = flag_face[flag_face['relative_position'] == orientation_check]
                        targets.append(np.stack([check_ids[tmp['face_id']],
                                                 flag_ids[tmp['flag_id']]], axis=1))
                    elif name == 'fd':
                        tmp = flag_data[(flag_data['relative_position_data'] == orientation_flag) &
                                        (flag_data['relative_position_face'] == orientation_check)]
                        targets.append(np.stack([flag_ids[tmp['flag_id']],
                                                 data_ids[tmp['data_id']]], axis=1))
                targets = np.concatenate(targets)
                ret.append(Instruction('CX', targets if basis == Pauli.X else targets[:, ::-1]))

            ret.append(Instruction('MX' if basis == Pauli.Z else 'M', qubit_ids['flag']))
            ret.append(Instruction('M' if basis == Pauli.Z else 'MX', qubit_ids['check']))
            all_check_ids.append(flag_face['flag_id'])
            relevant_checks = metadata.check[metadata.check['check_type'] == basis]
            all_check_ids.append(relevant_checks['check_id'])

        return SyndromeExtractionProcedure(
            instructions=ret,
            syndrome_ids=all_check_ids,
            syndrome_types=[SyndromeType.FLAG, SyndromeType.CHECK, SyndromeType.FLAG, SyndromeType.CHECK],
        )
