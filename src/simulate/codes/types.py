"""Data types and structures for quantum error correction codes.

This module contains the core data structures, enums, and type definitions
used throughout the quantum error correction code implementations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Iterable

import numpy as np

from ..instruction import Instruction
from ..measurement_tracker import SyndromeType


@dataclass
class CodeMetadata:
    """
    All codes must provide the following metadata (specified using numpy structured arrays)
    - tanner: np.ndarray with at least four columns 'data_id', 'check_id', 'pauli', 'edge_type',
              where 'pauli' just says what physical Pauli operator the check measures on the data qubit.
              The 'edge_type' is an integer that identifies the type of edge in the tanner graph.
              This type can be arbitrary and may only have meaning when the tanner graph has some
              structure or symmetry.
    - logical_operators: np.ndarray with at least one four columns 'logical_id', 'logical_pauli', 'data_id', 'physical_pauli',
                         where 'logical_id' is the index of the logical operator, 'logical_pauli' is the logical Pauli operator (X or Z)
                         and 'data_id', and 'physical_pauli' specify the support of the logical operator.
    - geometry: specifies the geometry of the code
    - extras: Dict[str, np.ndarray] of additional metadata.
    """
    tanner: np.ndarray
    logical_operators: np.ndarray
    check: np.ndarray
    data: np.ndarray
    entities: Dict[str, np.ndarray] = field(default_factory=dict) # Entities and their relations if they are many-to-one/one-to-many
    relations: Dict[str, np.ndarray] = field(default_factory=dict) # Other many-to-many relations

    _num_logical_qubits: Optional[int] = None

    def copy(self):
        return CodeMetadata(
            tanner=self.tanner.copy(),
            logical_operators=self.logical_operators.copy(),
            check=self.check.copy(),
            data=self.data.copy(),
            entities={k: v.copy() for k, v in self.entities.items()}
        )

    @property
    def num_data(self):
        return len(self.data)

    @property
    def num_checks(self):
        return len(self.check)
    
    @property
    def num_logical_qubits(self):
        if self._num_logical_qubits is None:
            self._num_logical_qubits = len(np.unique(self.logical_operators['logical_id']))
        return self._num_logical_qubits

    def __post_init__(self):
        self.tanner.setflags(write=False)
        self.logical_operators.setflags(write=False)
        self.check.setflags(write=False)
        self.data.setflags(write=False)
        if self.entities:
            for v in self.entities.values():
                v.setflags(write=False)


@dataclass
class SyndromeExtractionProcedure:
    """Data structure for syndrome extraction instructions and metadata.

    This standardizes the return type from get_syndrome_extraction_instructions
    across all quantum error correction codes.

    Attributes:
        instructions: List of Instructions for syndrome extraction
        targets: List of target qubit arrays for each measurement group
        types: List of measurement types ('data', 'check', 'flag')
        groups: List of qubit group names ('data', 'check', 'flag', etc.)
        roles: List of measurement roles (from MeasurementRole enum)
    """
    instructions: List[Instruction]
    # The first list indexes over distinct measurement groups (i.e., per line of M/MX stim instructions)
    # The second indexes over targets in those lines. The entries should be the check id (or ids) that 
    # that qubit belongs to.
    syndrome_ids: List[List[Union[int, Iterable[int]]]]
    syndrome_types: List[SyndromeType]
    data_ids: Optional[List[np.ndarray]] = None