"""CSS (Calderbank-Shor-Steane) quantum error correction codes.

This module provides the CSSCode class, which implements CSS codes that can
correct both X and Z errors independently using classical error correction codes.
"""

from typing import Optional

import numpy as np

from ..instruction import Instruction
from ..measurement_tracker import MeasurementTracker, SyndromeCoordinate, SyndromeType
from ..utils import groupby
from .base import Code
from ..types import Pauli


class CSSCode(Code):
    """Base class for CSS (Calderbank-Shor-Steane) codes.

    CSS codes are a class of quantum error correction codes that can correct
    both X and Z errors independently using classical error correction codes.
    They are defined by two commuting stabilizer groups, one for X-type errors
    and one for Z-type errors.
    """
    NAME = 'CSSCode'

    def _apply_cx(self, target: 'CSSCode', **_):
        """
        Applies a logical CNOT gate between this code (control) and a target code (target).

        Args:
            target: The target `CSSCode` instance.

        Returns:
            A list of `Instruction` objects representing the physical CX gates.
        """
        cx_targets = np.stack([self.qubit_ids['data'],
                              target.qubit_ids['data']], axis=1)
        return [Instruction('CX', cx_targets)]
    
    def check_ids(self, check_type: int) -> np.ndarray:
        """
        Returns the IDs of the check qubits for a given Pauli type (X or Z).

        Args:
            check_type: The Pauli type of the check (Pauli.X or Pauli.Z).

        Returns:
            A NumPy array of check IDs.
        """
        # The check entity must have column type 'check_type' for CSS codes
        metadata = self.metadata
        return metadata.check[metadata.check['check_type'] == check_type]['check_id']

    def _cx_detectors(self, meas_track: MeasurementTracker, target: 'CSSCode',
                     time: int):
        """
        Registers detectors for a logical CNOT gate between two codes.

        This method defines the detector patterns that fire when a logical CX
        gate is applied, relating measurements from previous and current time steps.

        Args:
            meas_track: The `MeasurementTracker` instance to register detectors with.
            target: The target `CSSCode` instance.
            time: The current time step of the simulation.
        """
        ctrl, targ = self.block_id, target.block_id
        x_check_ids = self.check_ids(Pauli.X)
        z_check_ids = self.check_ids(Pauli.Z)
        patterns = [
            [SyndromeCoordinate(time=time-1, block_id=ctrl, syndrome_type=SyndromeType.CHECK, ids=x_check_ids),
             SyndromeCoordinate(time=time-1, block_id=targ, syndrome_type=SyndromeType.CHECK, ids=x_check_ids),
             SyndromeCoordinate(time=time, block_id=ctrl, syndrome_type=SyndromeType.CHECK, ids=x_check_ids)],
            [SyndromeCoordinate(time=time-1, block_id=ctrl, syndrome_type=SyndromeType.CHECK, ids=z_check_ids),
             SyndromeCoordinate(time=time-1, block_id=targ, syndrome_type=SyndromeType.CHECK, ids=z_check_ids),
             SyndromeCoordinate(time=time, block_id=targ, syndrome_type=SyndromeType.CHECK, ids=z_check_ids)],
            [SyndromeCoordinate(time=time-1, block_id=ctrl, syndrome_type=SyndromeType.CHECK, ids=z_check_ids),
             SyndromeCoordinate(time=time, block_id=ctrl, syndrome_type=SyndromeType.CHECK, ids=z_check_ids)],
            [SyndromeCoordinate(time=time-1, block_id=targ, syndrome_type=SyndromeType.CHECK, ids=x_check_ids),
             SyndromeCoordinate(time=time, block_id=targ, syndrome_type=SyndromeType.CHECK, ids=x_check_ids)]
        ]
        for pattern in patterns:
            meas_track.register_detectors(pattern)

    def _apply_x(self, **_):
        """
        Applies a logical X gate to the data qubits.

        Returns:
            A list of `Instruction` objects representing the physical X gates.
        """
        return [Instruction('X', self.qubit_ids['data'])]

    def _apply_z(self, **_):
        """
        Applies a logical Z gate to the data qubits.

        Returns:
            A list of `Instruction` objects representing the physical Z gates.
        """
        return [Instruction('Z', self.qubit_ids['data'])]

    def _x_detectors(self, meas_track: MeasurementTracker, time: int):
        """
        Registers detectors for a logical X gate.

        For CSS codes, logical X primarily affects Z-type stabilizers, but
        this implementation defaults to `_i_detectors` for simplicity.

        Args:
            meas_track: The `MeasurementTracker` instance.
            time: The current time step.
        """
        self._i_detectors(meas_track, time)

    def _z_detectors(self, meas_track: MeasurementTracker, time: int):
        """
        Registers detectors for a logical Z gate.

        For CSS codes, logical Z primarily affects X-type stabilizers, but
        this implementation defaults to `_i_detectors` for simplicity.

        Args:
            meas_track: The `MeasurementTracker` instance.
            time: The current time step.
        """
        self._i_detectors(meas_track, time)

    def measure_data(self, meas_track: Optional[MeasurementTracker] = None,
                    basis: str = 'Z', **kwargs):
        """
        Measures all data qubits in the specified basis.

        Args:
            meas_track: Optional `MeasurementTracker` instance to register
                measurements with.
            basis: The measurement basis ('X' or 'Z'). Defaults to 'Z'.
            **kwargs: Additional keyword arguments, including 'time'.

        Returns:
            A list of `Instruction` objects representing the data measurements.
        """
        time = kwargs.get('time', -1)
        measurement_ids = None
        if meas_track is not None:
            check_ids = []
            pauli = Pauli.Z if basis == 'Z' else Pauli.X
            for group in groupby(self.metadata.tanner, by='data_id')[1]:
                check_ids.append(group[group['pauli'] == pauli]['check_id'])
            measurement_ids = meas_track.register_measurements(
                time=time,
                block_id=self.block_id,
                syndrome_type=SyndromeType.CHECK,
                syndrome_ids=check_ids,
                data_ids=np.arange(len(self.qubit_ids['data']))
            )
        instr_name = 'M' if basis == 'Z' else 'MX'
        meta = {'measurement_ids': measurement_ids}
        return [Instruction(instr_name, self.qubit_ids['data'], meta=meta)]

    def _m_detectors(self, meas_track: MeasurementTracker, time: int,
                    basis: str):
        """
        Registers detectors for data measurements (M or MX).

        Args:
            meas_track: The `MeasurementTracker` instance.
            time: The current time step.
            basis: The measurement basis ('X' or 'Z').
        """
        check_ids = self.check_ids(Pauli.Z if basis == 'Z' else Pauli.X)
        patterns = [SyndromeCoordinate(time=time-1, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=check_ids),
                    SyndromeCoordinate(time=time, block_id=self.block_id, syndrome_type=SyndromeType.CHECK, ids=check_ids)]
        meas_track.register_detectors(patterns)
