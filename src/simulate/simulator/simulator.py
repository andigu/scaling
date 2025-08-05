"""Core simulator interfaces for quantum error correction simulations.

Defines base classes for single-layer and multi-layer simulators, along with
standardized result structures for measurement tracking, detector outcomes,
and logical error detection.
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from ..algorithm import Algorithm
from ..codes import CodeFactory
from ..measurement_tracker import MeasurementTracker
from ..noise_model import NoiseModel

@dataclass
class SampleResult:
    """
    A dataclass to hold the results of a single quantum circuit simulation run.
    
    Attributes:
        measurement_tracker (MeasurementTracker): An instance of `MeasurementTracker`
            containing all recorded measurements and detector definitions.
        detectors (np.ndarray): A NumPy array of detector outcomes, typically
            binary (0 or 1), representing whether each detector fired.
            Shape: `(num_shots, num_detectors)` or `(num_time_steps, num_shots, num_detectors)`.
        logical_errors (np.ndarray): A NumPy array indicating whether a logical
            error occurred for each shot. Typically binary (0 or 1).
            Shape: `(num_shots, num_logical_qubits)` or `(num_time_steps, num_shots, num_logical_qubits)`.
        additional_data (Dict[str, Any]): A dictionary for any additional
            simulator-specific data, such as information about qubit losses.
    """
    measurement_tracker: MeasurementTracker
    detectors: np.ndarray
    logical_errors: np.ndarray
    additional_data: Dict[str, Any]

class BaseSimulator(ABC):
    """
    Abstract base class for all quantum circuit simulators.

    Provides common initialization and attributes for simulators, including
    the quantum algorithm, noise model, and code factory.
    """

    # Maximum value for Stim random seed (Stim uses a 63-bit integer for seeds)
    MAX_SEED_VALUE = 2 ** 63 - 1

    def __init__(self, algorithm: Algorithm, noise_model: NoiseModel,
                 code_factory: CodeFactory, seed: int = 0):
        """
        Initializes the base simulator.

        Args:
            algorithm: The `Algorithm` instance to be simulated.
            noise_model: The `NoiseModel` to apply during simulation.
            code_factory: The `CodeFactory` to create `Code` instances for logical qubits.
            seed: The random seed for the simulator's internal random number generator.
        """
        self.rng = np.random.default_rng(seed)
        self.algorithm = algorithm
        self.noise_model = noise_model
        self.code_factory = code_factory


class Simulator(BaseSimulator):
    """
    Abstract base class for single-layer (one-shot) quantum circuit simulators.

    Subclasses must implement the `sample` method to perform a simulation
    and return a single `SampleResult`.
    """

    @abstractmethod
    def sample(self, shots: int, **kwargs) -> SampleResult:
        """
        Samples from the quantum circuit simulation, returning a single result.

        Args:
            shots: The number of simulation shots to perform.
            **kwargs: Additional keyword arguments for the simulation.

        Returns:
            A `SampleResult` object containing the simulation outcomes.
        """
        pass


class LayeredSimulator(BaseSimulator):
    """
    Abstract base class for multi-layer quantum circuit simulators.

    Subclasses must implement the `sample` method to perform a simulation
    and return a list of `SampleResult` objects, one for each layer.
    """

    @abstractmethod
    def sample(self, shots: int, **kwargs) -> List[SampleResult]:
        """
        Samples from the quantum circuit simulation, returning results for each layer.

        Args:
            shots: The number of simulation shots to perform per layer.
            **kwargs: Additional keyword arguments for the simulation.

        Returns:
            A list of `SampleResult` objects, where each element corresponds
            to the outcome of a specific layer in the quantum algorithm.
        """
        pass
