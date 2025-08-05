from typing import List

from ..types import Pauli

from .decorators import supports_transversal_gates
from .css import CSSCode
from .steane_mixin import SteaneMixin
from .types import CodeMetadata
import pandas as pd
import numpy as np
from ..instruction import Instruction
from ..measurement_tracker import MeasurementTracker, SyndromeCoordinate, SyndromeType

@supports_transversal_gates(
    gates_1q=['I', 'X', 'Y', 'Z', 'S', 'S_DAG'],
    gates_2q=['CX', 'SWAP']
)
class QRMCode(SteaneMixin, CSSCode):
    """
    Implementation of the QRM(15, 1, 3) code.

    This code is a specific quantum Reed-Muller code with 15 physical qubits,
    1 logical qubit, and a distance of 3. It inherits from `SteaneMixin`
    for Steane-style syndrome extraction and `CSSCode` for CSS properties.
    """
    NAME = 'QRM(15, 1, 3)'

    @staticmethod
    def create_metadata(**_) -> CodeMetadata:
        """
        Creates the metadata for the QRM(15, 1, 3) code.

        This method hardcodes the Tanner graph (check-data qubit relationships),
        logical operators, and qubit metadata for the QRM(15, 1, 3) code.

        Returns:
            A `CodeMetadata` object containing the code's structural information.
        """
        x_checks = [
            "XXXXXXXXIIIIIII",
            "XXXXIIIIXXXXIII",
            "XXIIXXIIXXIIXXI",
            "XIXIXIXIXIXIXIX"
        ]
        z_checks = [
            "ZZZZZZZZIIIIIII",
            "ZZZZIIIIZZZZIII",
            "ZZIIZZIIZZIIZZI",
            "ZIZIZIZIZIZIZIZ",
            "ZZZZIIIIIIIIIII",
            "ZZIIZZIIIIIIIII",
            "ZIZIZIZIIIIIIII",
            "ZZIIIIIIZZIIIII",
            "ZIIIZIIIZIIIZII",
            "ZIZIIIIIZIZIIII"
        ]
        tanner = []
        checks = []
        for i, x_check in enumerate(x_checks):
            checks.append({'check_id': i, 'check_type': Pauli.X})
            for j, c in enumerate(x_check):
                if c == 'X':
                    tanner.append({
                        'data_id': j,
                        'check_id': i,
                        'pauli': Pauli.X,
                        'edge_type': 0
                    })
        for i, z_check in enumerate(z_checks):
            checks.append({'check_id': i + len(x_checks), 'check_type': Pauli.Z})
            for j, c in enumerate(z_check):
                if c == 'Z':
                    tanner.append({
                        'data_id': j,
                        'check_id': i + len(x_checks),
                        'pauli': Pauli.Z,
                        'edge_type': 0
                    })
        tanner = pd.DataFrame(tanner).reset_index().to_records(index=False)
        logical_operators = []
        for p in [Pauli.X, Pauli.Z]:
            for i in range(12 if p == Pauli.Z else 8, 15):
                logical_operators.append({
                    'logical_id': 0,
                    'logical_pauli': p,
                    'data_id': i,
                    'physical_pauli': p
                })
        logical_operators = pd.DataFrame(logical_operators).to_records(index=False)
        check = pd.DataFrame(checks).to_records(index=False)
        data = pd.DataFrame(np.unique(tanner['data_id']), columns=['data_id']).to_records(index=False)
        return CodeMetadata(
            tanner=tanner,
            logical_operators=logical_operators,
            check=check,
            data=data,
        )

    def _apply_s(self, **_) -> List[Instruction]:
        """
        Applies a logical S gate to the data qubits.

        Returns:
            A list containing a single `S` instruction on the data qubits.
        """
        return [Instruction('S', self.qubit_ids['data'])]

    def _apply_s_dag(self, **_) -> List[Instruction]:
        """
        Applies a logical S_DAG gate to the data qubits.

        Returns:
            A list containing a single `S_DAG` instruction on the data qubits.
        """
        return [Instruction('S_DAG', self.qubit_ids['data'])]
    
    def _apply_y(self, **_) -> List[Instruction]:
        """
        Applies a logical Y gate to the data qubits.

        Returns:
            A list containing a single `Y` instruction on the data qubits.
        """
        return [Instruction('Y', self.qubit_ids['data'])]
    
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
    
    def _s_detectors(self, meas_track: MeasurementTracker, time: int):
        """
        Registers detectors for a logical S gate.

        For QRM, this involves specific correlations between X and Z check
        measurements across time steps.

        Args:
            meas_track: The `MeasurementTracker` instance.
            time: The current time step.
        """
        x_check_ids, z_check_ids = self.check_ids(Pauli.X), self.check_ids(Pauli.Z)
        n_x = len(x_check_ids)
        patterns = [
            [SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=x_check_ids),
             SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=z_check_ids[:n_x]),
             SyndromeCoordinate(time=time, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=x_check_ids)],
            [SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=z_check_ids),
             SyndromeCoordinate(time=time, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=z_check_ids)]
        ]
        for pattern in patterns:
            meas_track.register_detectors(pattern)
    
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
