"""Logical error calculation utilities for quantum error correction simulations."""

from typing import List
import numpy as np

from ..algorithm import Algorithm
from ..codes import Code
from ..types import Pauli
from ..measurement_tracker import MeasurementTracker


class LogicalErrorCalculator:
    """Calculates logical errors from physical errors or measurement outcomes.
    
    This class encapsulates the complex logic for determining logical errors
    based on quantum error correction code structure and observable operators.
    """

    @staticmethod
    def from_physical_errors(
        measurement_basis: np.ndarray,
        x_err: np.ndarray,
        z_err: np.ndarray,
        logical_qubits: List[Code]
    ) -> np.ndarray:
        """Calculate logical errors from physical X and Z errors.
        
        Args:
            algorithm: The quantum algorithm being simulated
            x_err: Physical X errors for each shot and qubit
            z_err: Physical Z errors for each shot and qubit  
            logical_qubits: List of quantum error correction codes
            
        Returns:
            Boolean array of logical errors for each shot and logical qubit
        """
        shots = x_err.shape[0]
        n = len(measurement_basis)
        n_per_block = logical_qubits[0].num_logical_qubits
        if n_per_block > 1:
            logical_errors = np.zeros((shots, n, n_per_block), dtype=bool)    
        else:
            logical_errors = np.zeros((shots, n), dtype=bool)

        for i, basis in enumerate(measurement_basis):
            code = logical_qubits[i]
            los = code.metadata.logical_operators
            for j in range(n_per_block):
                data_idxs = los[(los['logical_pauli'] == basis) & (los['logical_id'] == j)]['data_id']
                relevant_qubits = code.qubit_ids['data'][data_idxs]
                err = (np.sum(x_err[:, relevant_qubits], axis=-1) % 2).astype(bool) if basis == Pauli.Z else (np.sum(z_err[:, relevant_qubits], axis=-1) % 2).astype(bool)
                if n_per_block > 1:
                    logical_errors[:, i, j] = err
                else:
                    logical_errors[:, i] = err

        return logical_errors

    @staticmethod
    def from_measurements(
        algorithm: Algorithm,
        meas_sim: np.ndarray,
        logical_qubits: List[Code],
        meas_tracker: MeasurementTracker,
        deterministic_output_signs: np.ndarray
    ) -> np.ndarray:
        """Calculate logical errors from measurement outcomes.
        
        Args:
            algorithm: The quantum algorithm being simulated
            meas_sim: Measurement outcomes for each shot
            logical_qubits: List of quantum error correction codes
            meas_tracker: Tracks measurement positions and metadata
            deterministic_output_signs: Expected output signs for logical measurements
            
        Returns:
            Boolean array of logical errors for each shot and logical qubit
        """
        shots = meas_sim.shape[0]
        n_per_block = logical_qubits[0].num_logical_qubits
        if n_per_block > 1:
            logical_errors = np.zeros((shots, algorithm.n, n_per_block), dtype=bool)    
        else:
            logical_errors = np.zeros((shots, algorithm.n), dtype=bool)
        # Get final data measurements
        measurements = meas_tracker.measurements
        final_measurements = measurements[
            (measurements['time'] == measurements['time'].max())
        ]

        for i, basis in enumerate(algorithm.get_measurement_basis()):
            code = logical_qubits[i]
            for j in range(n_per_block):
                data_idxs = code.metadata.logical_operators[
                    (code.metadata.logical_operators['logical_pauli'] == basis) &
                    (code.metadata.logical_operators['logical_id'] == j)
                ]['data_id']
                # Find measurements corresponding to relevant qubits for this logical qubit
                relevant_measurements = final_measurements[
                    (final_measurements['block_id'] == code.block_id) &
                    np.isin(final_measurements['data_id'], data_idxs)
                ]['measurement_position']
                
                # Calculate logical measurement outcome and compare to expected
                logical_outcome = np.sum(meas_sim[:, relevant_measurements], axis=-1) % 2
                if n_per_block > 1:
                    logical_errors[:, i, j] = (logical_outcome == deterministic_output_signs[i]).astype(bool)
                else:
                    logical_errors[:, i] = (logical_outcome == deterministic_output_signs[i]).astype(bool)

        return logical_errors
