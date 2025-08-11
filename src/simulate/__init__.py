"""Quantum error correction simulation package.

This package provides tools for simulating quantum error correction codes,
including surface codes and color codes, with realistic noise models and
measurement tracking capabilities.
"""

from .algorithm import Algorithm, Layer
from .codes import (
    Code, CSSCode, CodeFactory, ColorCode, SurfaceCode, SteaneSurfaceCode,
    SurfaceCodeBase, QRMCode, BivariateBicycle
)
from .instruction import Instruction
from .logical_circuit import LogicalCircuit
from .measurement_tracker import MeasurementTracker, SyndromeType, MeasurementRole, SyndromeCoordinate
from .noise_model import NoiseModel
from .simulator import StimSimulator, LayeredStimSimulator, SampleResult
from .types import Pauli

__all__ = [
    # Core algorithms and circuits
    'Algorithm',
    'Layer',
    'Instruction',
    'LogicalCircuit',
    
    # Quantum error correction codes
    'Code',
    'CSSCode',
    'CodeFactory',
    'SurfaceCode',
    'SurfaceCodeBase',
    'ColorCode',
    'QRMCode',
    'SteaneSurfaceCode',
    'BivariateBicycle',
    
    # Measurement tracking
    'MeasurementTracker',
    'SyndromeType',
    'MeasurementRole',
    'SyndromeCoordinate',
    
    # Noise modeling
    'NoiseModel',
    
    # Simulation
    'StimSimulator',
    'LayeredStimSimulator',
    'SampleResult',

    'Pauli'
]
