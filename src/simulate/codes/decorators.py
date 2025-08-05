"""Decorators for quantum error correction codes.

This module contains decorators used to enhance and validate quantum error
correction code implementations.
"""

from typing import List, Optional


def supports_transversal_gates(*, gates_1q: Optional[List[str]] = None,
                              gates_2q: Optional[List[str]] = None):
    """Class decorator to declare transversal gate support.

    This decorator automatically:
    1. Sets the TRANSVERSAL_1Q and TRANSVERSAL_2Q class attributes
    2. Validates that required methods exist

    Args:
        gates_1q: List of single-qubit transversal gates (e.g., ['H', 'X'])
        gates_2q: List of two-qubit transversal gates (e.g., ['CX', 'SWAP'])

    Usage:
        @supports_transversal_gates(gates_1q=['H', 'X'], gates_2q=['CX'])
        class MyCode(Code):
            # Must implement _apply_h, _h_detectors, etc.
            pass
    """
    def class_decorator(cls):
        # Set the transversal gate lists as class attributes
        cls.TRANSVERSAL_1Q = gates_1q or []
        cls.TRANSVERSAL_2Q = gates_2q or []

        # Validate immediately at class definition time to catch errors early
        # Design rationale: This enforces the interface contract at import time
        # rather than at runtime, making debugging easier and ensuring completeness
        all_gates = cls.TRANSVERSAL_1Q + cls.TRANSVERSAL_2Q
        for gate_name in all_gates:
            # Each transversal gate requires two methods:
            # 1. _apply_X: Returns Instructions for implementing the gate
            # 2. _X_detectors: Adds appropriate detectors for error detection
            apply_method = f'_apply_{gate_name.lower()}'
            detector_method = f'_{gate_name.lower()}_detectors'

            # Check that both required methods exist
            if not hasattr(cls, apply_method):
                error_msg = (f"Code {cls.__name__} claims to support "
                           f"transversal gate '{gate_name}' but is missing "
                           f"required method '{apply_method}'. Each transversal gate "
                           f"must implement both application and detector methods.")
                raise NotImplementedError(error_msg)
            if not hasattr(cls, detector_method):
                error_msg = (f"Code {cls.__name__} claims to support "
                           f"transversal gate '{gate_name}' but is missing "
                           f"required method '{detector_method}'. Each transversal gate "
                           f"must implement both application and detector methods.")
                raise NotImplementedError(error_msg)
        return cls

    return class_decorator