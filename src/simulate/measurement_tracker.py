"""Measurement tracking for quantum error correction circuits.

This module provides classes for tracking measurements and detectors in
quantum error correction simulations, including the MeasurementRole
enumeration and MeasurementTracker class.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union, Iterable
import warnings

import numpy as np
import scipy.sparse as sp

class SyndromeType(IntEnum):
    CHECK = 0
    FLAG = 1

class MeasurementRole(IntEnum):
    X_CHECK = 0
    Z_CHECK = 1
    X_FLAG = 2
    Z_FLAG = 3
    DATA_X = 4
    DATA_Z = 5

@dataclass
class SyndromeCoordinate:
    """
    Represents a coordinate for a syndrome measurement, uniquely identifying it
    by time, logical block, syndrome type, and ID within that type.
    """
    time: int
    block_id: int
    syndrome_type: SyndromeType
    ids: np.ndarray # Array of syndrome IDs (e.g., check IDs, flag IDs)

    @property
    def key(self) -> Tuple[int, int, SyndromeType]:
        """
        Returns a unique key for the syndrome coordinate, useful for dictionary lookup.
        """
        return (self.time, self.block_id, self.syndrome_type)

class MeasurementTracker:
    """Tracks measurements and detectors for quantum error correction codes.

    This class manages collections of measurements and detectors, and provides
    methods for updating them and calculating logical errors. It uses structured
    NumPy arrays for efficient storage and querying.
    """

    def __init__(self):
        # Define measurement dtype using structured NumPy arrays
        # This enables efficient querying and filtering by any combination of fields
        # Structured arrays provide both memory efficiency and 
        # convenient access patterns for ML data preprocessing
        measurement_dtype = [
            ('measurement_id', int),      # Unique identifier for each measurement
            ('time', int),                # Circuit layer when measurement occurs
            ('block_id', int),    # Which logical qubit this measurement belongs to
            ('measurement_position', int), # Position in global measurement sequence
            ('data_id', int), # -1 if the measured qubit is not a data qubit, 
                              # otherwise the index of the data qubit that was measured
            ('subcycle', int) # Syndrome extraction rounds can have subcycles, 
                             # or 'mini-rounds' within the round (e.g., color code or Steane extraction)
        ]
        self.measurements = np.empty(0, dtype=measurement_dtype)

        # Define detector dtype for Stim detector construction
        # Detectors group related measurements for syndrome extraction
        detector_dtype = [
            ('detector_id', int),         # Unique detector identifier
            ('time', int),                # When detector fires
            ('block_id', int),    # Associated logical qubit
            ('syndrome_type', int),                # Type of detector (stabilizer, flag, etc.)
            ('syndrome_id', int)
        ]
        self.detectors = np.empty(0, dtype=detector_dtype)

        # Measurement-to-detector mapping for syndrome construction
        # This bidirectional mapping enables efficient conversion between
        # individual measurements and grouped detector outcomes
        # measurement_position will be arbitrary until measurements are
        # finalized (see LogicalCircuit.attach_detectors)
        self.m2d = np.empty(0, dtype=[('measurement_id', int),
                                      ('detector_id', int)])

        # Dict[(time, qubit_id, role)][check_id] = measurement IDs for check
        self.syndrome_to_meas: Dict[Tuple[int, int, SyndromeType], Dict[int, List[int]]] = {}
        self._all_measurement_positions_set = False

    def set_measurement_positions(self, measurement_positions: np.ndarray):
        """
        Sets the global measurement positions for all recorded measurements.

        This method is typically called after all instructions have been added
        to a `LogicalCircuit` and their final positions are determined.
        It's crucial for constructing the measurement-to-detector (m2d) matrix.

        Args:
            measurement_positions: A 1D NumPy array where `measurement_positions[i]`
                is the global index of the i-th measurement recorded by the tracker.
        """
        self.measurements['measurement_position'] = measurement_positions
        self._all_measurement_positions_set = True

    @property
    def m2d_matrix(self) -> sp.csr_matrix:
        """
        Returns the measurement-to-detector matrix as a sparse SciPy CSR matrix.

        This matrix maps individual measurements to their corresponding detector
        outcomes. Each row corresponds to a detector, and each column to a measurement.
        A non-zero entry indicates that a specific measurement contributes to a detector.

        Raises:
            UserWarning: If `set_measurement_positions` has not been called,
                         indicating that measurement positions might be arbitrary.

        Returns:
            A sparse CSR matrix of shape `(num_detectors, num_measurements)`.
        """
        if not self._all_measurement_positions_set:
            warnings.warn('Measurement positions have not all been set, this may return garbage.')
        meas = np.sort(self.measurements, order=['measurement_id'])
        measurement_positions = meas[self.m2d['measurement_id']]
        measurement_positions = measurement_positions['measurement_position']

        ret = sp.csr_matrix(
            (np.ones(self.m2d.shape[0]),
             (self.m2d['detector_id'], measurement_positions)),
            shape=(self.detectors.shape[0], self.measurements.shape[0]),
            dtype=np.uint8
        )
        return ret

    def register_measurements(self, time: int,
                             block_id: int, 
                             syndrome_type: SyndromeType,
                             syndrome_ids: List[Union[int, Iterable[int]]],
                             data_ids: Optional[Iterable[int]] = None,
                             subcycle: int = 0) -> np.ndarray:
        """
        Registers a set of measurements for a logical qubit at a specific time.

        This method records metadata about each measurement, including its time,
        associated logical block, type, and the physical qubits involved.

        Args:
            time: The time step (circuit layer) when these measurements occur.
            block_id: The unique identifier of the logical qubit block.
            syndrome_type: The `SyndromeType` (e.g., CHECK, FLAG) this measurement
                           is associated with.
            syndrome_ids: A list of IDs for the syndromes being measured. This can be
                          a list of single integers or iterables of integers if multiple
                          physical checks correspond to one logical syndrome.
            data_ids: Optional. An iterable of physical data qubit IDs involved in
                      each measurement. If None, indicates a non-data qubit measurement.
            subcycle: The sub-round index within a larger syndrome extraction round,
                      useful for codes with multi-step measurement protocols.

        Returns:
            A NumPy array of newly assigned unique measurement IDs.
            
        Raises:
            ValueError: If input parameters are invalid (e.g., negative time/block_id).
        """
        # Input validation
        if time < 0:
            raise ValueError(f'Invalid time step {time}: must be non-negative')
        if block_id < 0:
            raise ValueError(f'Invalid block_id {block_id}: must be non-negative')
        n_measurements = len(syndrome_ids)
        start_id = self.measurements.shape[0]
        new_measurement_ids = np.arange(start_id, start_id + n_measurements)
        to_add = np.zeros(n_measurements, dtype=self.measurements.dtype)
        to_add['measurement_id'] = new_measurement_ids
        to_add['time'] = time
        to_add['block_id'] = block_id
        if data_ids is not None:
            to_add['data_id'] = data_ids
        else:
            to_add['data_id'] = -1
        to_add['subcycle'] = subcycle
        self.measurements = np.concatenate([self.measurements, to_add], axis=0)
        
        key = (time, block_id, syndrome_type)
        for i, ids in enumerate(syndrome_ids):
            if isinstance(ids, (int, np.integer)):
                ids = [ids]
            for check_id in ids:
                if key not in self.syndrome_to_meas:
                    self.syndrome_to_meas[key] = {}
                if check_id in self.syndrome_to_meas[key]:
                    self.syndrome_to_meas[key][check_id].append(new_measurement_ids[i])
                else:
                    self.syndrome_to_meas[key][check_id] = [new_measurement_ids[i]]
        return new_measurement_ids

    def register_detectors(self, syndrome_coordinates: List[SyndromeCoordinate]) -> List[int]:
        """
        Registers a set of detectors based on provided syndrome coordinates.

        Detectors are logical constructs that fire when an odd number of associated
        measurements flip. This method links registered measurements to new detectors.

        Args:
            syndrome_coordinates: A list of `SyndromeCoordinate` objects. Each
                                  `SyndromeCoordinate` specifies the time, block ID,
                                  syndrome type, and an array of syndrome IDs for
                                  a group of measurements that define a detector.

        Returns:
            A list of newly assigned unique detector IDs.

        Raises:
            ValueError: If `syndrome_coordinates` is empty, or if `ids` arrays
                        within `SyndromeCoordinate` objects have inconsistent lengths,
                        or if a syndrome ID is not found in `syndrome_to_meas`.
        """
        if len(syndrome_coordinates) == 0:
            return []  # Nothing to register
        if any(len(x.ids) != len(syndrome_coordinates[0].ids) for x in syndrome_coordinates):
            raise ValueError('Syndrome coordinates must have the same number of IDs')
        measurement_ids = []
        n_detectors = len(syndrome_coordinates[0].ids)
        detector_id = self.detectors.shape[0]
        detector_ids, times, block_ids, syndrome_types, syndrome_ids = [], [], [], [], []
        for i in range(n_detectors):
            measurement_ids.append([])
            is_present = False
            for coordinate in syndrome_coordinates:
                if coordinate.key in self.syndrome_to_meas:
                    if coordinate.ids[i] in self.syndrome_to_meas[coordinate.key]:
                        measurement_ids[-1].extend(self.syndrome_to_meas[coordinate.key][coordinate.ids[i]])
                        is_present = True
                    else:
                        raise ValueError(f'Syndrome id {coordinate.ids[i]} not found in syndrome_to_meas for key {coordinate.key}')
            if is_present:
                times.append(syndrome_coordinates[-1].time)
                block_ids.append(syndrome_coordinates[-1].block_id)
                syndrome_types.append(syndrome_coordinates[-1].syndrome_type)
                syndrome_ids.append(syndrome_coordinates[-1].ids[i])
                detector_ids.append(detector_id)
                detector_id += 1
        to_add = np.zeros(len(detector_ids), dtype=self.detectors.dtype)
        to_add['time'] = times
        to_add['detector_id'] = detector_ids
        to_add['block_id'] = block_ids
        to_add['syndrome_type'] = syndrome_types
        to_add['syndrome_id'] = syndrome_ids
        self.detectors = np.concatenate([self.detectors, to_add], axis=0)
        
        # Build measurement-to-detector mapping
        m2d_update = []
        for i, meas_ids in enumerate(measurement_ids):
            for measurement_id in meas_ids:
                m2d_update.append((measurement_id, detector_ids[i]))

        if m2d_update:
            m2d_array = np.array(m2d_update, dtype=self.m2d.dtype)
            self.m2d = np.concatenate([self.m2d, m2d_array], axis=0)
        return detector_ids

    def copy(self) -> 'MeasurementTracker':
        """Create a deep copy of this MeasurementTracker."""
        ret = MeasurementTracker()
        ret.measurements = self.measurements.copy()
        ret.detectors = self.detectors.copy()
        ret.m2d = self.m2d.copy()
        ret.syndrome_to_meas = {k: {k2: v2.copy() for k2, v2 in v.items()} for k, v in self.syndrome_to_meas.items()}
        ret._all_measurement_positions_set = self._all_measurement_positions_set # pylint: disable=protected-access
        return ret

    def __repr__(self) -> str:
        return (f'<MeasurementTracker with {self.measurements.shape[0]} '
                f'measurements and {self.detectors.shape[0]} detectors>')
