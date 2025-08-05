"""Quantum instruction representation and manipulation.

This module provides the Instruction class for representing quantum operations
in circuits, including gates, measurements, and noise operations. Instructions
can be converted to different formats and manipulated for circuit construction.
"""

import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np

from . import utils


class Instruction:
    """Represents a single quantum instruction in a circuit.
    
    An Instruction encapsulates a quantum operation with its name, target qubits,
    optional arguments, and metadata. Instructions can represent gates, measurements,
    noise operations, and custom operations for quantum error correction.
    
    Attributes:
        name: Name of the instruction (e.g., 'H', 'CX', 'M')
        targets: Array of target qubit indices
        args: Optional arguments for parameterized operations
        meta: Dictionary of metadata for the instruction
    """
    # Barrier separates layers of the circuit and is a meta instruction for compilation
    CUSTOM_INSTRUCTIONS = set(['LOSS', 'LINE_REF', 'BARRIER'])
    NONCLIFFORD_GATES = set(['T'])
    TWO_QUBIT_GATES = set(['CX', 'SWAP'])
    TWO_QUBIT_NOISE = set(['DEPOLARIZE2'])
    SINGLE_QUBIT_NOISE = set(['DEPOLARIZE1', 'X_ERROR', 'Z_ERROR', 'LOSS'])
    SINGLE_QUBIT_GATES = set(['I', 'H', 'S', 'S_DAG', 'X', 'Y', 'Z', 'SQRT_Y', 'SQRT_Y_DAG', 'T'])
    MEASURE = set(['M', 'MX', 'MR', 'MRX'])
    RESET = set(['R', 'RX', 'MR', 'MRX'])
    # Map instruction names to unique integer IDs
    INSTRUCTION_TO_ID = {
        # Single qubit gates
        'I': 0,
        'H': 1,
        'S': 2,
        'S_DAG': 3,
        'X': 4,
        'Y': 5,
        'Z': 6,
        'SQRT_Y': 7,
        'SQRT_Y_DAG': 8,
        'T': 9,

        # Two qubit gates
        'CX': 10,
        'SWAP': 11,

        # Single qubit noise
        'DEPOLARIZE1': 12,
        'X_ERROR': 13,
        'Z_ERROR': 14,

        # Two qubit noise
        'DEPOLARIZE2': 15,

        # Custom instructions
        'LOSS': 16,
        'LINE_REF': 17,
        'BARRIER': 18,

        # Other instructions seen in codebase
        'DETECTOR': 19,
        'MPP': 20,
        'M': 21,
        'MX': 22,
        'MR': 23,
        'MRX': 24,
        'R': 25,
        'RX': 26
    }

    # Reverse mapping from ID to instruction name
    ID_TO_INSTRUCTION = {v: k for k, v in INSTRUCTION_TO_ID.items()}

    @classmethod
    def name_to_id(cls, name: str) -> int:
        """Convert instruction name to integer ID"""
        return cls.INSTRUCTION_TO_ID[name]

    @classmethod
    def id_to_name(cls, iden: int) -> str:
        """Convert integer ID back to instruction name"""
        return cls.ID_TO_INSTRUCTION[iden]

    @property
    def id(self) -> int:
        """Get the integer ID for this instruction"""
        return self.name_to_id(self.name)

    def __init__(self, name: str, targets: np.ndarray,
                 args: Optional[Union[float, List[float]]] = None,
                 meta: Optional[Dict[str, Any]] = None):
        self.name = name
        self.targets: np.ndarray = targets
        self.args = args
        self.meta = meta if meta is not None else {}
        self._str_cache = None

    @property
    def is_unitary(self) -> bool:
        """Check if this instruction represents a unitary operation."""
        return self.name in Instruction.SINGLE_QUBIT_GATES.union(
            Instruction.TWO_QUBIT_GATES)
    
    def inverse(self) -> 'Instruction':
        """Return the inverse of this instruction.
        
        Returns:
            Instruction representing the inverse operation
            
        Raises:
            ValueError: If the instruction cannot be inverted
        """
        if self.name in ['I', 'H', 'X', 'Y', 'Z', 'SWAP']:
            return self.copy()
        elif self.name == 'CX':
            ret = self.copy()
            # Not required in theory, but strange problem in stim with CX
            ret.targets = ret.targets[::-1] 
            return ret
        elif self.name == 'S':
            ret = self.copy()
            ret.name = 'S_DAG'
            return ret
        elif self.name == 'S_DAG':
            ret = self.copy()
            ret.name = 'S'
            return ret
        else:
            raise ValueError(f'Cannot invert instruction {self.name}')
   
    def copy(self, target_remapper: Optional[np.ndarray] = None) -> 'Instruction':
        """Create a deep copy of this instruction."""
        return Instruction(self.name, self.targets.copy() if target_remapper is None else target_remapper[self.targets],
                          copy.deepcopy(self.args), copy.deepcopy(self.meta))

    def __str__(self) -> str:
        """Convert instruction to string representation."""
        if self._str_cache is not None:
            return self._str_cache
        if self.name == 'BARRIER':
            ret = f'==================== BARRIER({self.meta['type']}) ===================='
        elif self.name == 'DETECTOR':
            ret = 'DETECTOR ' + ' '.join(f'rec[{int(x)}]' for x in self.flattened_targets)
        else:
            target_str = ' '.join(map(str, self.flattened_targets))
            if self.args is None:
                ret = self.name + ' ' + target_str
            else:
                if isinstance(self.args, (float, int)):
                    arg_str = str(self.args)
                else:
                    arg_str = ', '.join(map(str, self.args))
                ret = f'{self.name}({arg_str}) ' + target_str
        self._str_cache = ret
        return ret

    def __repr__(self) -> str:
        """Return string representation of the instruction."""
        return str(self)

    def remove_qubits(self, losses: np.ndarray) -> 'Instruction':
        """Create a new instruction with specified qubits removed.
        
        Args:
            losses: Boolean array indicating which qubits to remove
            
        Returns:
            New Instruction with the specified qubits removed
        """
        if self.name == 'DETECTOR':
            return self
        else:
            return Instruction(self.name, self.targets[~losses], self.args)
    
    @property
    def flattened_targets(self) -> List[int]:
        """Get flattened list of all target qubits."""
        return utils.flatten(self.targets)
