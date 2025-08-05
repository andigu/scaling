"""Factory classes for creating quantum error correction code instances.

This module provides factory classes that manage resource allocation and
instance creation for quantum error correction codes.
"""

from typing import Any, Dict, Generic, Optional, Type, TypeVar

from .base import Code

T = TypeVar('T', bound=Code)

class CodeFactory(Generic[T]):
    """Factory for creating Code instances with automatic resource management.

    Manages physical qubit allocation and logical qubit ID assignment
    across multiple code instances.
    """

    def __init__(self, code_class: Type[T], code_params: Optional[Dict[str, Any]] = None):
        """Initialize the factory.

        Args:
            code_class: Class of code to create instances of
            code_params: Parameters to pass to code constructor
        """
        self.code_class = code_class
        self.code_params = code_params or {}
        self.index_start = 0
        self.block_id = 0

    def __call__(self) -> T:
        """Create a new Code instance with unique resource allocation.

        Returns:
            New Code instance with allocated qubits and ID
        """
        ret = self.code_class.from_code_params(
            **self.code_params,
            physical_id_start=self.index_start,
            block_id=self.block_id
        )
        self.index_start += ret.num_qubits
        self.block_id += 1
        return ret
