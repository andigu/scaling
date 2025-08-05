"""Quantum algorithm representation and manipulation.

This module provides classes for representing quantum algorithms as sequences
of gate layers and converting them to quantum circuits. It includes support
for random algorithm generation and circuit manipulation.
"""

from collections import UserDict, UserList
from typing import List, Self, Optional, Union, Tuple, Dict

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from .types import Pauli

class Layer(UserDict):
    """A layer of quantum gates organized by gate type.

    This class extends defaultdict to organize quantum gates by their type,
    where each gate type maps to a list of qubit indices.
    """
    def __init__(self, data: Optional[Dict[str, Union[List, np.ndarray]]] = None):
        if data is None or len(data) == 0:
            super().__init__()
            return
        data_np = {k: np.array(v) for k, v in data.items()}
        targets = np.concatenate([v.flatten() for v in data_np.values()])
        if len(set(targets)) != len(targets):
            raise ValueError(f'Conflicting gate assignments in layer: {targets}. '
                             f'Each qubit can only have one gate per layer.')
        super().__init__(data_np)

    @property
    def n(self) -> int:
        return max([v.max(initial=-1) for v in self.values()]) + 1

    def copy(self):
        return Layer({k: v.copy() for k, v in self.items()})
    
    def add_gate(self, gate: str, target: Union[int, Tuple[int, int]]) -> None:
        if gate not in self:
            self[gate] = np.array([target], dtype=int)
        else:
            self[gate] = np.concatenate([self[gate], np.array([target], dtype=int)])

    def __setitem__(self, key: str, value: Union[List, np.ndarray]) -> None:
        super().__setitem__(key, np.array(value))
    
    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self:
            return np.array([], dtype=int)
        return super().__getitem__(key)
    
    @property
    def two_qubit_gates(self) -> 'Layer':
        return Layer({k: v for k, v in self.items() if len(v.shape) == 2})

    @property
    def single_qubit_gates(self) -> 'Layer':
        return Layer({k: v for k, v in self.items() if len(v.shape) == 1 and k not in ['M', 'MX']})

    @staticmethod
    def single_qubit_layer(gates: list[str]) -> 'Layer':
        # gates = ['I', 'X', 'Y'] gives {"I": [0], "X": [1], "Y": [2]}
        ret = {}
        for i, gate in enumerate(gates):
            if gate not in ret: ret[gate] = [i]
            else: ret[gate].append(i)
        return Layer(ret)

class Algorithm(UserList[Layer]):
    """Represents a quantum algorithm as a sequence of gate layers.

    An Algorithm consists of a sequence of gate layers that can be converted 
    to quantum circuits. Each layer contains gates organized by type.

    Attributes:
        n: Number of qubits in the algorithm (computed from layers).
    """

    @property
    def n(self) -> int:
        return max([layer.n for layer in self])
    
    def get_measurement_basis(self, time: Optional[int] = None) -> np.ndarray:
        relevant_layer = self[-1 if time is None else time]
        ret = [-1 for _ in range(self.n)]
        for gate_type, qubit_indices in relevant_layer.items():
            if gate_type in ['M', 'MX']:
                for qubit_idx in qubit_indices:
                    ret[qubit_idx] = Pauli.Z if gate_type == 'M' else Pauli.X
        return np.array(ret)

    def combine(self, other: Self):
        """Combines two algorithms independently.

        The second algorithm's qubits are shifted by self.n to avoid conflicts.
        Both algorithms must have compatible layer structures.

        Args:
            other: Algorithm to combine with this one

        Returns:
            New Algorithm containing both combined algorithms
        """
        # Create a new algorithm with combined number of qubits
        result = Algorithm()

        # Combine layers
        max_layers = max(len(self), len(other))
        for i in range(max_layers):
            combined_layer = Layer()

            # Add first algorithm's layer if it exists
            if i < len(self):
                combined_layer = self[i].copy()

            # Add second algorithm's layer with shifted indices
            if i < len(other):
                for gate_type, qubit_indices in other[i].items():
                    combined_layer[gate_type] = np.concatenate([combined_layer[gate_type], qubit_indices + self.n])

            # Add the combined layer to the result
            result.append(combined_layer)

        return result

    def copy(self) -> 'Algorithm':
        ret = Algorithm()
        ret.extend([l.copy() for l in self])
        return ret

    @staticmethod
    def build_memory(basis=Pauli.Z, cycles=2):
        algorithm = Algorithm()
        for _ in range(cycles):
            algorithm.append(Layer({'I': [0]}))
        algorithm.append(Layer({'M' if basis == Pauli.Z else 'MX': [0]}))
        return algorithm

    def set_name(self, name: str) -> Self:
        self.name = name
        return self

    @staticmethod
    def build_random(n, rng, single_qubit_gates, depth=4,
                     entangle_prob=0.5, swap_prob=0.2):
        """
        Build a random quantum algorithm.

        Args:
            n: Number of qubits
            rng: Random number generator
            single_qubit_gates: List of available single-qubit gates
            depth: Number of circuit layers
            entangle_prob: Probability of choosing a two-qubit gate instead
                of single-qubit gates
            swap_prob: Given a two-qubit gate is chosen, probability of it
                being a SWAP (vs CX)
        """
        algorithm = Algorithm()
        for _ in range(depth):
            # For each layer, initialize empty gate lists
            layer = Layer()

            # Track which qubits are available (not yet used in this layer)
            available_qubits = set(range(n))

            # While we have at least 2 qubits available
            while len(available_qubits) >= 2:
                # Decide if we want to do an entangling operation
                if rng.random() < entangle_prob and len(available_qubits) >= 2:
                    # Choose 2 qubits for the entangling operation
                    q1, q2 = rng.choice(list(available_qubits), size=2, replace=False)
                    available_qubits.remove(q1)
                    available_qubits.remove(q2)

                    # Choose between CX and SWAP
                    if rng.random() < swap_prob:
                        layer.add_gate('SWAP', (q1, q2))
                    else:
                        layer.add_gate('CX', (q1, q2))
                else:
                    # Apply a single-qubit gate to one qubit
                    q = rng.choice(list(available_qubits))
                    available_qubits.remove(q)
                    gate = rng.choice(single_qubit_gates)
                    layer.add_gate(gate, q)

            # Handle any remaining single qubit
            if available_qubits:
                q = list(available_qubits)[0]
                gate = rng.choice(single_qubit_gates)
                layer.add_gate(gate, q)
            # Add the constructed layer
            algorithm.append(layer)

        # Add final measurement layer
        meas_basis = rng.choice(['X', 'Z'], size=n)
        mx = [i for i in range(n) if meas_basis[i] == 'X']
        mz = [i for i in range(n) if meas_basis[i] == 'Z']
        algorithm.append(Layer({'MX': mx, 'M': mz}))

        return algorithm

    def to_qiskit(self):
        n = self.n
        qr = QuantumRegister(n)
        cr = ClassicalRegister(n)
        qc = QuantumCircuit(qr, cr)
        tmp = []
        qc.barrier()
        for layer in self:
            for i in layer['X']:
                qc.x(qr[i])
            for i in layer['Y']:
                qc.y(qr[i])
            for i in layer['Z']:
                qc.z(qr[i])
            for i in layer['H']:
                qc.h(qr[i])
            for i in layer['S']:
                qc.s(qr[i])
            for (ctrl, targ) in layer['CX']:
                qc.cx(qr[ctrl], qr[targ])
            for (ctrl, targ) in layer['SWAP']:
                qc.swap(qr[ctrl], qr[targ])
            for i in layer['M']:
                qc.measure(qr[i], cr[i])
            for i in layer['MX']:
                qc.h(qr[i])
                qc.measure(qr[i], cr[i])
            qc.barrier()
        return qc

    @staticmethod
    def from_qiskit(qc):
        n = len(qc.qubits)
        ret = Algorithm()
        for layer in circuit_to_dag(qc).layers():
            circ = dag_to_circuit(layer['graph'])
            l = Layer()
            for gate in circ.data:
                name = gate.operation.name.upper()
                if len(gate.qubits) == 1:
                    idx = circ.qubits.index(gate.qubits[0])
                    l.add_gate(name, idx)
                if len(gate.qubits) == 2:
                    ctrl, targ = [circ.qubits.index(x) for x in gate.qubits]
                    l.add_gate(name, (ctrl, targ))
            ret.append(l)
        ret.append(Layer({'M': np.arange(n)}))
        return ret
