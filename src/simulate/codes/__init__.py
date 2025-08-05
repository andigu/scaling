"""Quantum error correction codes module.

This module provides implementations of various quantum error correction
codes including surface codes and color codes.
"""

from .base import Code
from .css import CSSCode
from .decorators import supports_transversal_gates
from .factory import CodeFactory
from .steane_mixin import SteaneMixin
from .types import CodeMetadata, SyndromeExtractionProcedure

# from .bb import BivariateBicycle
from .color import ColorCode
from .surface import SurfaceCode, SteaneSurfaceCode, SurfaceCodeBase
from .qrm import QRMCode

__all__ = [
    # Base classes
    'Code', 'CSSCode', 'CodeFactory',
    # Types and decorators
    'CodeMetadata', 'SyndromeExtractionProcedure', 'supports_transversal_gates',
    # Mixins and decorators
    'SteaneMixin',
    # Specific code implementations
    'ColorCode', 'SurfaceCode', 'SteaneSurfaceCode', 'SurfaceCodeBase', 'QRMCode',
    # Utilities
    # 'Visualizer'
]
