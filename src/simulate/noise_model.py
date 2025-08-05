"""Noise models for quantum error correction simulations.

This module provides noise models that can be applied to quantum circuits
to simulate realistic quantum hardware behavior, including gate errors,
measurement errors, and qubit loss.
"""

from typing import Any, Dict, List, Union

import numpy as np

from .instruction import Instruction
from .logical_circuit import LogicalCircuit
from .utils import flatten


class NoiseModel:
    """
    Represents a comprehensive noise model for quantum error correction simulations.
    
    This class allows for the application of various noise mechanisms to quantum
    circuits, simulating realistic hardware imperfections such as gate errors,
    measurement errors, reset errors, and qubit loss.
    """
    DEFAULT_NOISE_PARAMS: Dict[str, float] = dict(
        measurement_error_rate=5e-3,
        measurement_loss_rate=5e-3,
        reset_error_rate=5e-3,
        reset_loss_rate=5e-3,
        single_qubit_error_rate=5e-4,
        single_qubit_loss_rate=5e-4,
        idle_error_rate=1e-4,
        two_qubit_error_rate=5e-3,
        two_qubit_loss_rate=5e-3,
    )

    def __init__(self, **noise_params: float) -> None:
        """
        Initializes the noise model with specified parameters.
        
        Parameters not provided will default to 0 (no noise for that parameter).

        Args:
            **noise_params: Keyword arguments specifying the noise rates.
                            Valid keys correspond to the attributes of `NoiseModel`.
        """
        self.measurement_error_rate = noise_params.get(
            'measurement_error_rate', 0)
        self.measurement_loss_rate = noise_params.get(
            'measurement_loss_rate', 0)
        self.reset_error_rate = noise_params.get('reset_error_rate', 0)
        self.reset_loss_rate = noise_params.get('reset_loss_rate', 0)
        self.single_qubit_error_rate = noise_params.get(
            'single_qubit_error_rate', 0)
        self.single_qubit_loss_rate = noise_params.get(
            'single_qubit_loss_rate', 0)
        self.two_qubit_error_rate = noise_params.get('two_qubit_error_rate', 0)
        self.two_qubit_loss_rate = noise_params.get('two_qubit_loss_rate', 0)
        self.idle_error_rate = noise_params.get('idle_error_rate', 0)

    def fine_rescale(self, rng: np.random.Generator, lo: float = 0.8,
                     hi: float = 1.2) -> 'NoiseModel':
        """
        Creates a new `NoiseModel` with each parameter randomly rescaled.
        
        Each noise parameter in the current model is multiplied by a random
        factor sampled uniformly from `[lo, hi]`.

        Args:
            rng: A NumPy `Generator` for random number generation.
            lo: The lower bound for the uniform scaling factor.
            hi: The upper bound for the uniform scaling factor.
            
        Returns:
            A new `NoiseModel` instance with rescaled parameters.
        """
        # scale = 10**rng.uniform(np.log10(0.1), np.log10(max_scale),
        #                         size=len(NoiseModel.DEFAULT_NOISE_PARAMS))
        scale = rng.uniform(
            lo, hi, size=len(NoiseModel.DEFAULT_NOISE_PARAMS))
        return NoiseModel(
            **{k: v * scale[i] for i, (k, v) in enumerate(vars(self).items())})

    def scale_loss(self, scale: float) -> 'NoiseModel':
        """
        Scales only the loss-related noise parameters by a given factor.
        
        All parameters containing 'loss' in their name will be multiplied by `scale`.

        Args:
            scale: The scaling factor for loss rates.
            
        Returns:
            A new `NoiseModel` instance with scaled loss parameters.
        """
        return NoiseModel(
            **{k: v * (scale if 'loss' in k else 1)
               for k, v in vars(self).items()})

    def scale_pauli(self, scale: float) -> 'NoiseModel':
        """
        Scales only the Pauli error rates by a given factor.
        
        All parameters containing 'error' in their name (and not 'loss')
        will be multiplied by `scale`.

        Args:
            scale: The scaling factor for Pauli error rates.
            
        Returns:
            A new `NoiseModel` instance with scaled error parameters.
        """
        return NoiseModel(
            **{k: v * (scale if 'error' in k else 1)
               for k, v in vars(self).items()})

    def __repr__(self) -> str:
        """Return string representation of noise parameters."""
        return str(vars(self))

    @property
    def has_loss(self) -> bool:
        """
        Checks if this noise model includes any loss mechanisms.

        Returns:
            True if any of the loss rates (`measurement_loss_rate`,
            `reset_loss_rate`, `single_qubit_loss_rate`, `two_qubit_loss_rate`)
            are non-zero, False otherwise.
        """
        return any([
            self.measurement_loss_rate, self.reset_loss_rate,
            self.single_qubit_loss_rate, self.two_qubit_loss_rate
        ])

    @staticmethod
    def scale_noise(params: Dict[str, Any], scale: float) -> Dict[str, Any]:
        """
        Scales noise parameters by a constant factor.
        
        Args:
            params: A dictionary of noise parameters (e.g., from `DEFAULT_NOISE_PARAMS`).
            scale: The scaling factor.
            
        Returns:
            A new dictionary with the scaled parameters.
        """
        return {
            k: (tuple([v2 * scale for v2 in v]) if isinstance(v, tuple)
                else v * scale)
            for k, v in params.items()
        }
    @staticmethod
    def get_scaled_noise_model(scale: float = 1) -> 'NoiseModel':
        """
        Creates a `NoiseModel` with scaled default parameters.
        
        Args:
            scale: The scaling factor to apply to the `DEFAULT_NOISE_PARAMS`.
            
        Returns:
            A `NoiseModel` instance with scaled parameters.
        """
        return NoiseModel(
            **NoiseModel.scale_noise(NoiseModel.DEFAULT_NOISE_PARAMS, scale))

    def without_loss(self) -> 'NoiseModel':
        """
        Creates a new `NoiseModel` instance with all loss rates set to zero.
        
        Returns:
            A new `NoiseModel` instance representing a lossless version of the current model.
        """
        return NoiseModel(**dict(
            measurement_error_rate=self.measurement_error_rate,
            measurement_loss_rate=0,
            reset_error_rate=self.reset_error_rate,
            reset_loss_rate=0,
            single_qubit_error_rate=self.single_qubit_error_rate,
            single_qubit_loss_rate=0,
            two_qubit_error_rate=self.two_qubit_error_rate,
            two_qubit_loss_rate=0,
            idle_error_rate=self.idle_error_rate
        ))

    @staticmethod
    def from_array(arr: np.ndarray) -> 'NoiseModel':
        """
        Creates a `NoiseModel` instance from a NumPy array of parameters.
        
        The input array must contain the noise parameters in a predefined order.

        Args:
            arr: A 1D NumPy array of noise parameters.
            
        Returns:
            A `NoiseModel` instance constructed from the array.
        """
        noise_params = dict(
            measurement_error_rate=arr[0],
            measurement_loss_rate=arr[1],
            reset_error_rate=arr[2],
            reset_loss_rate=arr[3],
            single_qubit_error_rate=arr[4],
            single_qubit_loss_rate=arr[5],
            two_qubit_error_rate=arr[6],
            two_qubit_loss_rate=arr[7],
            idle_error_rate=arr[8]
        )
        return NoiseModel(**noise_params)

    def to_array(self, include_loss: bool = True) -> np.ndarray:
        """
        Converts the `NoiseModel` parameters to a 9-dimensional NumPy array.
        
        Returns:
            A 1D NumPy array of noise parameters (always includes loss parameters).
        """
        if include_loss:
            return np.array([
                self.measurement_error_rate, self.measurement_loss_rate,
                self.reset_error_rate, self.reset_loss_rate,
                self.single_qubit_error_rate, self.single_qubit_loss_rate,
                self.two_qubit_error_rate, self.two_qubit_loss_rate,
                self.idle_error_rate
                ], dtype=np.float32)
        else:
            return np.array([
                self.measurement_error_rate, self.reset_error_rate,
                self.single_qubit_error_rate, self.two_qubit_error_rate,
                self.idle_error_rate
            ], dtype=np.float32)

    def __call__(self, op: Union[Instruction, LogicalCircuit]
                ) -> LogicalCircuit:
        """
        Applies the noise model to a quantum instruction or a `LogicalCircuit`.
        
        This method injects Stim-compatible noise instructions (e.g., DEPOLARIZE, LOSS, X_ERROR)
        into the circuit based on the configured noise rates.

        Args:
            op: The `Instruction` or `LogicalCircuit` to which noise should be applied.
            
        Returns:
            A new `LogicalCircuit` containing the original operations with added noise.
            
        Raises:
            NotImplementedError: If the gate type of an instruction is not supported
                                 by the noise model.
        """
        if isinstance(op, LogicalCircuit):
            instr: List[Instruction] = []
            for x in op:
                instr.extend(self(x))
            return LogicalCircuit(instr)
        else:
            op_name, targets, _ = op.name, op.targets, op.args
            if (op_name in ['MPP', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'BARRIER'] or
                    ('noiseless' in op.meta and op.meta['noiseless'])):
                ret = [op]
            elif op_name == 'CZ' or op_name == 'CX':
                ret = [op]
                if self.two_qubit_error_rate:
                    ret.append(Instruction(
                        'DEPOLARIZE2', targets, self.two_qubit_error_rate))
                if self.two_qubit_loss_rate:
                    ret.append(Instruction('LOSS', targets, self.two_qubit_loss_rate))
            elif op_name == 'SWAP':
                ret = [op]
                targs = np.array(flatten(targets))
                if self.single_qubit_error_rate:
                    ret.append(Instruction(
                        'DEPOLARIZE1', targs, self.single_qubit_error_rate))
                if self.single_qubit_loss_rate:
                    ret.append(Instruction('LOSS', targs, self.single_qubit_loss_rate))
            else:
                match op_name:
                    case 'I':
                        ret = [op, ]
                        if self.idle_error_rate:
                            ret.append(Instruction(
                                'DEPOLARIZE1', targets, self.idle_error_rate))
                    case 'H' | 'S' | 'S_DAG' | 'X' | 'Y' | 'Z' | 'SQRT_Y' | 'SQRT_Y_DAG' | 'T':
                        ret = [op, ]
                        if self.single_qubit_error_rate:
                            ret.append(Instruction('DEPOLARIZE1', targets,
                                                self.single_qubit_error_rate))
                        if self.single_qubit_loss_rate:
                            ret.append(Instruction('LOSS', targets, self.single_qubit_loss_rate))
                    case 'MR' | 'MRX':
                        ret = []
                        if self.measurement_loss_rate:
                            ret.append(Instruction('LOSS', targets, self.measurement_loss_rate))
                        if self.measurement_error_rate:
                            ret.append(Instruction('X_ERROR' if op_name == 'MR' else 'Z_ERROR', targets,
                                                self.measurement_error_rate))
                        ret.append(op)
                        if self.reset_error_rate:
                            ret.append(
                                Instruction('X_ERROR' if op_name == 'MR' else 'Z_ERROR',
                                        targets, self.reset_error_rate))
                        if self.reset_loss_rate:
                            ret.append(Instruction('LOSS', targets, self.reset_loss_rate))
                    case 'R' | 'RX':
                        ret = [op]
                        if self.reset_error_rate:
                            ret.append(
                                Instruction('X_ERROR' if op_name == 'R' else 'Z_ERROR',
                                        targets, self.reset_error_rate))
                        if self.reset_loss_rate:
                            ret.append(Instruction('LOSS', targets, self.reset_loss_rate))
                    case 'M' | 'MX':
                        ret = []
                        if self.measurement_loss_rate:
                            ret.append(Instruction('LOSS', targets, self.measurement_loss_rate))
                        if self.measurement_error_rate:
                            ret.append(Instruction('X_ERROR' if op_name == 'M' else 'Z_ERROR', targets,
                                                self.measurement_error_rate))
                        ret.append(op)
                    case _:
                        raise NotImplementedError('No known gate noise for ' + op_name)
            return LogicalCircuit(ret)
