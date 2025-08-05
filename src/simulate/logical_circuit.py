"""Logical quantum circuit representation and manipulation.

This module provides the LogicalCircuit class for representing quantum circuits
as sequences of instructions, with support for circuit construction from
algorithms, noise application, and conversion to simulation formats.
"""

from collections import UserList
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import stim

from . import utils
from .algorithm import Algorithm, Layer
from .codes import Code, CodeFactory
from .instruction import Instruction
from .measurement_tracker import MeasurementTracker
from .types import Pauli


class LogicalCircuit(UserList[Instruction]):
    """
    Represents a logical quantum circuit composed of quantum instructions.

    A LogicalCircuit is a container for quantum circuit instructions that can
    be converted to a stim Circuit for simulation. It provides methods for
    constructing circuits from algorithms and performing various operations.
    """

    def __init__(self, instructions: Optional[Iterable[Instruction]] = None):
        """
        Initialize a LogicalCircuit with optional instructions.

        Args:
            instructions: Collection of instructions to include in the circuit
        """
        super().__init__(instructions or [])

    @property
    def n_qubits(self) -> int:
        """
        Calculate the number of qubits used in this circuit.

        Returns:
            The maximum qubit index used in the circuit plus one
        """
        # Filter out instructions with string targets (e.g. DETECTOR instructions)
        # Find max qubit index used and add 1 (as qubit indices are 0-based)
        return max([np.max(x.targets, initial=-1) for x in self.data], default=-1) + 1

    def to_stim(self) -> stim.Circuit:
        """
        Convert the logical circuit to a stim Circuit for simulation.

        Returns:
            A stim.Circuit object representing this logical circuit
        """
        return stim.Circuit('\n'.join(map(str, self.data)))

    @staticmethod
    def build_unitary_layer(
            layer: Layer,
            qubits: List[Code],
            meas_tracker: MeasurementTracker,
            time: int,
    ) -> 'LogicalCircuit':
        """
        Build a logical circuit for a single layer of unitary operations.

        Args:
            layer: Dictionary mapping gate types to lists of target qubits
            qubits: List of Code objects representing logical qubits
            meas_tracker: MeasurementTracker to update with measurements
            time: Current time step in the circuit

        Returns:
            LogicalCircuit containing the instructions for this layer
        """
        circuit = LogicalCircuit()
        operated_qubits = []  # Track which qubits had operations applied
        stabilizer_measurement_instructions = []

        # Process SWAP operations
        for gate, targets in layer.two_qubit_gates.items():
            for (first_idx, second_idx) in targets:
                operated_qubits.extend([first_idx, second_idx])
                first_qubit, second_qubit = qubits[first_idx], qubits[second_idx]
                circuit.extend(first_qubit.apply_gate(gate=gate, target=second_qubit,
                                                     time=time))
                # Measure stabilizers for both qubits
                stabilizer_measurement_instructions.append(
                    first_qubit.measure_stabilizers(meas_track=meas_tracker, time=time)
                )
                stabilizer_measurement_instructions.append(
                    second_qubit.measure_stabilizers(meas_track=meas_tracker, time=time)
                )
                first_qubit.construct_detectors(meas_track=meas_tracker, gate=gate,
                                               time=time, target=second_qubit)

        # Process Pauli operations (I, X, Y, Z)
        for gate, targets in layer.single_qubit_gates.items():
            for targ in targets:
                operated_qubits.append(targ)
                qubit = qubits[targ]
                circuit.extend(qubit.apply_gate(gate, time=time))
                stabilizer_measurement_instructions.append(
                    qubit.measure_stabilizers(meas_track=meas_tracker, time=time)
                )
                qubit.construct_detectors(meas_track=meas_tracker, gate=gate, time=time)
        # Add all stabilizer measurements
        if stabilizer_measurement_instructions:
            circuit.extend(LogicalCircuit.combine_instructions(
                stabilizer_measurement_instructions))

        return circuit

    @staticmethod
    def build_measure_layer(
            layer: Layer,
            qubits: List[Code],
            meas_tracker: MeasurementTracker,
            time: int,
    ) -> 'LogicalCircuit':
        """
        Build a circuit layer for measuring logical qubits.

        Args:
            layer: Dictionary with 'M' and/or 'MX' keys mapping to lists of qubit indices
            qubits: List of Code objects representing logical qubits
            meas_tracker: MeasurementTracker to update with measurements
            time: Current time step in the circuit

        Returns:
            LogicalCircuit containing measurement instructions
        """
        circuit = LogicalCircuit()
        for basis_key, basis in [('M', 'Z'), ('MX', 'X')]:
            for qubit_idx in layer.get(basis_key, []):
                # Measure qubit data in specified basis
                instructions = qubits[qubit_idx].measure_data(
                    meas_tracker,
                    basis=basis,
                    add_observable=False,
                    time=time,
                )
                circuit.extend(instructions)
                qubits[qubit_idx].construct_detectors(meas_track=meas_tracker, gate='M',
                                                     time=time, basis=basis)
        return circuit

    def get_detector_circuit(self, meas_track: MeasurementTracker):
        measurement_positions = np.zeros(meas_track.measurements.shape[0], dtype=int)
        n_meas = 0
        for instr in self:
            if instr.name in ['M', 'MX', 'MR', 'MRX']:
                measurement_positions[instr.meta['measurement_ids']] = np.arange(
                    n_meas, n_meas + len(instr.targets))
                n_meas += len(instr.targets)
        m2d = np.sort(meas_track.m2d, order=['detector_id', 'measurement_id'])
        meas_track.set_measurement_positions(measurement_positions)

        _, unique_indices = np.unique(m2d['detector_id'], return_index=True)
        return LogicalCircuit([
            Instruction('DETECTOR',
                       measurement_positions[measurements_group['measurement_id']] - n_meas)
            for _, measurements_group in enumerate(np.split(m2d, unique_indices[1:]))
        ])

    @staticmethod
    def build_init_layer(
            algorithm: Algorithm,
            out_signs: Iterable[bool],
            qubits: List[Code]
    ) -> 'LogicalCircuit':
        """
        Build the initialization layer of a logical circuit from an algorithm.

        When the algorithm is run and transversal readouts are done, the outcomes
        are deterministic and equal to out_signs (in the noiseless case).
        """
        basis = algorithm.get_measurement_basis()
        circuit = LogicalCircuit()
        for i, basis, sign in zip(range(algorithm.n), basis, out_signs):
            circuit.extend(qubits[i].initialize(basis=basis, signs=bool(sign)))
        for layer in reversed(algorithm[:-1]):
            for gate, targets in layer.two_qubit_gates.items():
                instrs = []
                for (ctrl, targ) in targets:
                    gate_instructions = qubits[ctrl].apply_gate(gate, target=qubits[targ])
                    instrs.append([x.inverse() for x in gate_instructions])
                circuit.extend(LogicalCircuit.combine_instructions(instrs))
            for gate, targets in layer.single_qubit_gates.items():
                instrs = []
                for targ in targets:
                    instrs.append([x.inverse() for x in qubits[targ].apply_gate(gate)])
                circuit.extend(LogicalCircuit.combine_instructions(instrs))
        for instr in circuit:
            instr.meta['noiseless'] = True
        return circuit

    def inverse(self) -> 'LogicalCircuit':
        """
        Invert the logical circuit.
        """
        return LogicalCircuit([x.inverse() for x in reversed(self)])

    @staticmethod
    def from_algorithm(
            algorithm: Algorithm,
            logical_qubit_factory: CodeFactory,
            deterministic_output_signs: Optional[Union[List[bool], np.ndarray]] = None,
    ) -> Tuple['LogicalCircuit', List[Code], MeasurementTracker]:
        """
        Create a complete logical circuit from an algorithm.

        Args:
            algorithm: Algorithm to convert to a logical circuit
            code_factory: Function that creates Code objects for each logical qubit

        Returns:
            A tuple containing:
              - The complete logical circuit
              - List of Code objects representing the logical qubits
              - MeasurementTracker with all tracked measurements
        """
        logical_qubits = [logical_qubit_factory()
                          for _ in range(algorithm.n)]
        circuit = LogicalCircuit()
        meas_tracker = MeasurementTracker()
        time = 0
        # Initialize qubits
        if deterministic_output_signs is not None:
            app = LogicalCircuit.build_init_layer(
                algorithm, deterministic_output_signs, logical_qubits)
            circuit.extend(app)
        else:
            instrs = []
            for i in range(algorithm.n):
                instrs.append(logical_qubits[i].initialize(basis=Pauli.Z, signs=True))
            circuit.extend(LogicalCircuit.combine_instructions(instrs))

        # Process each layer of the algorithm except the final measurement layer
        for layer in algorithm[:-1]:
            time += 1
            layer_circuit = LogicalCircuit.build_unitary_layer(
                layer, logical_qubits, meas_tracker, time)
            circuit.extend(layer_circuit)

        # Process final measurement layer
        time += 1
        measure_layer = LogicalCircuit.build_measure_layer(
            algorithm[-1], logical_qubits, meas_tracker, time)
        circuit.extend(measure_layer)
        circuit.extend(circuit.get_detector_circuit(meas_tracker))
        return circuit, logical_qubits, meas_tracker

    @staticmethod
    def from_algorithm_layered(
            algorithm: Algorithm,
            logical_qubit_factory: List[Code] | CodeFactory,
            deterministic_output_signs: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[List['LogicalCircuit'], List['LogicalCircuit'], List['LogicalCircuit']],
    List[Code],
    List[MeasurementTracker]]:
        """
        Create a layered logical circuit representation from an algorithm.

        This separates the circuit into initialization, unitary layers, and measurement layers
        to enable layer-by-layer simulation.

        Args:
            algorithm: Algorithm to convert to a layered logical circuit
            code_factory: Function that creates Code objects for each logical qubit

        Returns:
            A tuple containing:
              - A tuple of (initialization circuits, unitary layers, measurement layers)
              - List of Code objects representing the logical qubits
              - List of MeasurementTrackers for each measurement layer
        """
        logical_qubits = []
        if isinstance(logical_qubit_factory, List):
            logical_qubits = logical_qubit_factory
        else:
            logical_qubits = [logical_qubit_factory() for _ in range(algorithm.n)]
        # Set up measurement tracking
        num_logical_qubits = len(logical_qubits)
        meas_tracker = MeasurementTracker()
        time = 0

        # Create initialization circuit
        init_circuits = []
        unitary_layers = []
        measurement_layers = []
        meas_trackers = []
        current_alg = algorithm.copy()
        current_alg.clear()
        current_alg.append(algorithm[-1])
        cumulative_unitary = LogicalCircuit()
        # Process each layer of the algorithm except the final measurement layer
        for layer in algorithm[:-1]:
            time += 1
            current_alg.insert(len(current_alg) - 1, layer)
            if deterministic_output_signs is None:
                instrs = []
                for i in range(num_logical_qubits):
                    instrs.append(logical_qubits[i].initialize(basis=Pauli.Z, signs=True))
                init_circuits.append(LogicalCircuit.combine_instructions(instrs))
            else:
                init_layer = LogicalCircuit.build_init_layer(
                    current_alg, deterministic_output_signs, logical_qubits)
                init_circuits.append(init_layer)

            # For each unitary layer, create a corresponding measurement layer
            # with a deep copy of the tracker to enable independent simulation
            unitary_layer = LogicalCircuit.build_unitary_layer(
                layer, logical_qubits, meas_tracker, time)
            cumulative_unitary.extend(unitary_layer)
            unitary_layers.append(unitary_layer)

            meas_tracker_copy = meas_tracker.copy()
            meas_layer = LogicalCircuit.build_measure_layer(
                algorithm[-1],
                logical_qubits,  # No need to do second deepcopy of the qubits
                meas_tracker_copy,
                time + 1
            )
            combined_circuit = cumulative_unitary + meas_layer
            detector_circuit = combined_circuit.get_detector_circuit(meas_tracker_copy)
            meas_layer.extend(detector_circuit)
            measurement_layers.append(meas_layer)
            meas_trackers.append(meas_tracker_copy)
            
        
        return (init_circuits, unitary_layers, measurement_layers), logical_qubits, meas_trackers

    @staticmethod
    def combine_instructions(instruction_sets: List[List[Instruction]]) -> 'LogicalCircuit':
        """
        Combine multiple sets of instructions into a single list, merging instructions at each time.

        This creates combined instructions where targets from all instruction sets are merged.
        Instructions are assumed to be aligned by time step across all instruction sets.

        Args:
            instruction_sets: List of instruction lists to combine

        Returns:
            Combined list of instructions
        """
        if not instruction_sets:
            return LogicalCircuit()

        result = LogicalCircuit()
        # Process each time step
        for t in range(len(instruction_sets[0])):
            # Get properties from first instruction at this time
            # (All instructions at this time should have same name/args/meta)
            name = instruction_sets[0][t].name
            args = instruction_sets[0][t].args

            # Verify that all instructions at this time have same properties
            for instruction_set in instruction_sets:
                if instruction_set[t].name != name:
                    raise ValueError(
                        f'Instruction names don\'t match: expected "{name}", '
                        f'got "{instruction_set[t].name}"')
                if instruction_set[t].args != args:
                    raise ValueError(
                        f'Instruction arguments don\'t match: expected {args}, '
                        f'got {instruction_set[t].args}')

            # Combine targets from all instruction sets
            all_targets = []
            current_targets = set()
            meta = {}
            for instruction_set in instruction_sets:
                # Check for target conflicts
                new_targets = set(utils.flatten(instruction_set[t].targets))
                conflicts = current_targets.intersection(new_targets)
                if len(conflicts) > 0:
                    raise ValueError(
                        f'Target conflict detected: qubits {list(conflicts)} '
                        f'are used in multiple instruction sets')
                current_targets = current_targets.union(new_targets)
                all_targets.extend(instruction_set[t].targets)
                measurement_ids = instruction_set[t].meta.get('measurement_ids', None)
                if measurement_ids is not None:
                    if 'measurement_ids' not in meta:
                        meta['measurement_ids'] = measurement_ids
                    else:
                        meta['measurement_ids'] = np.concatenate(
                            [meta['measurement_ids'], measurement_ids])
                if 'noiseless' in instruction_set[t].meta:
                    meta['noiseless'] = True
                if meta.get('noiseless', False) != instruction_set[t].meta.get('noiseless', False):
                    raise ValueError('Noisy and noiseless instructions cannot be combined')
            # Only create combined instruction if there are targets
            if len(all_targets) > 0:
                result.append(Instruction(name, np.array(all_targets),
                                         args=args, meta=meta))

        return result

    @property
    def is_clifford(self) -> bool:
        """
        Check if the circuit contains only Clifford gates (no T gates).

        Returns:
            True if the circuit is a Clifford circuit, False otherwise
        """
        return not any(instr.name == 'T' for instr in self)

    def without_noise(self) -> 'LogicalCircuit':
        """
        Create a copy of this circuit with all noise instructions removed.

        Returns:
            A new LogicalCircuit without noise instructions
        """
        noise_instructions = Instruction.SINGLE_QUBIT_NOISE.union(
            Instruction.TWO_QUBIT_NOISE)
        return LogicalCircuit(filter(
            lambda op: op.name not in noise_instructions,
            self
        ))

    def without_custom_instructions(self) -> 'LogicalCircuit':
        """
        Create a copy of this circuit with all custom instructions removed.

        Returns:
            A new LogicalCircuit without custom instructions
        """
        return LogicalCircuit(filter(
            lambda op: op.name not in Instruction.CUSTOM_INSTRUCTIONS,
            self
        ))

    def __repr__(self) -> str:
        """Return a formatted representation of the circuit with line numbers."""
        ndigits = len(str(len(self)))
        return '\n'.join([f'{(i+1):{ndigits}d} | {str(x)}' for i, x in enumerate(self)])


    def __str__(self) -> str:
        """Convert the circuit to a string."""
        return f'<Circuit with {len(self)} instructions>'
