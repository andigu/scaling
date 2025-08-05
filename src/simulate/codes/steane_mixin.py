"""Decorators and mixins for quantum error correction codes.

This module provides reusable decorators and mixins that add common functionality
to quantum error correction code implementations.
"""

from collections import OrderedDict

import numpy as np

from ..types import Pauli

from ..instruction import Instruction
from ..measurement_tracker import SyndromeType
from ..utils import groupby
from .css import CSSCode
from .types import SyndromeExtractionProcedure

class SteaneMixin(CSSCode):
    @classmethod
    def get_qubit_ids(cls, index_start: int = 0, **code_params) -> OrderedDict[str, np.ndarray]:
        """Get qubit IDs for Steane-style syndrome extraction.
        
        Automatically allocates data qubits and teleport auxiliary qubits.
        The number of teleport qubits equals the number of data qubits.
        
        Args:
            index_start: Starting index for qubit allocation
            **code_params: Parameters passed to create_metadata
            
        Returns:
            OrderedDict with 'data' and 'teleport' qubit arrays
        """
        metadata = cls.create_metadata(**code_params)
        num_data = len(metadata.data)
        return OrderedDict([
            ('data', np.arange(index_start, index_start + num_data)),
            ('teleport', np.arange(index_start + num_data, index_start + 2 * num_data))
        ])
 
    @classmethod
    def get_syndrome_extraction_instructions(cls, **code_params) -> SyndromeExtractionProcedure:
        """Generate Steane-style syndrome extraction instructions.
        
        Args:
            **code_params: Parameters for the code (passed to create_metadata)
            
        Returns:
            SyndromeExtractionProcedure with teleportation-based syndrome extraction
        """
        
        ret = []
        qubits = cls.get_qubit_ids(**code_params)
        teleport, data = qubits['teleport'], qubits['data']
        metadata = cls.create_metadata(**code_params)
        
        # Initialize auxiliary qubits in X basis (for first Bell measurement)
        x_initialize = [
            Instruction(instr.name, teleport[instr.targets], meta={'noiseless': True})
            for instr in cls._initialize(paulis=(Pauli.X,), signs=(True,), **code_params)
        ]
        
        # Initialize data qubits in Z basis (for second Bell measurement)  
        z_initialize = [
            Instruction(instr.name, data[instr.targets], meta={'noiseless': True})
            for instr in cls._initialize(paulis=(Pauli.Z,), signs=(True,), **code_params)
        ]

        # First Bell measurement cycle - measure Z syndrome
        ret.extend(x_initialize)
        ret.append(Instruction('CX', np.stack([teleport, data], axis=1)))
        ret.append(Instruction('CX', np.stack([data, teleport], axis=1)))
        ret.append(Instruction('M', data))

        # Second Bell measurement cycle - measure X syndrome  
        ret.extend(z_initialize)
        ret.append(Instruction('CX', np.stack([teleport, data], axis=1)))
        ret.append(Instruction('CX', np.stack([data, teleport], axis=1)))
        ret.append(Instruction('MX', teleport))
        
        # Extract syndrome IDs from tanner graph structure
        z_check_ids = []
        x_check_ids = []
        for group in groupby(metadata.tanner, by='data_id')[1]:
            z_check_ids.append(group[group['pauli'] == Pauli.Z]['check_id'].tolist())
            x_check_ids.append(group[group['pauli'] == Pauli.X]['check_id'].tolist())
            
        return SyndromeExtractionProcedure(
            instructions=ret,
            syndrome_ids=[z_check_ids, x_check_ids],
            syndrome_types=[SyndromeType.CHECK, SyndromeType.CHECK],
            data_ids=[np.arange(len(data)), np.arange(len(teleport))]
        )
    