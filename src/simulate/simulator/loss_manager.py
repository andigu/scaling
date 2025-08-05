import numpy as np
from typing import List, Optional, Tuple
import copy
import stim
from .. import utils
from ..instruction import Instruction
from ..logical_circuit import LogicalCircuit

class LossManager:
    def __init__(self, n_qubits: int, rng: np.random.Generator, batch_size: int, current_losses: Optional[np.ndarray] = None):
        self.n_qubits = n_qubits
        self.rng = rng
        self.batch_size = batch_size
        self.current_losses = np.zeros((batch_size, n_qubits), dtype=bool) if current_losses is None else current_losses
    
    def copy(self):
        return LossManager(self.n_qubits, copy.deepcopy(self.rng), self.batch_size, self.current_losses.copy())

    def loss_approximation(self, fs: stim.FlipSimulator, circuit: LogicalCircuit):
        shots = fs.batch_size
        loss_probs = np.array([op.args for op in circuit if op.name == 'LOSS'])
        n_targs = np.array([len(op.targets) for op in circuit if op.name == 'LOSS'])
        rng1, rng2 = self.rng.spawn(2)
        if len(loss_probs) > 0:
            n_losses = iter(rng1.binomial(n_targs * shots, loss_probs))
        else:
            n_losses = iter([])
        measurement_mask = []
        cumulative_circuit_str = ''
        for i, op in enumerate(circuit):
            if op.name == 'LOSS':
                n_loss = next(n_losses)
                if n_loss > 0:
                    targs = np.array(utils.flatten(op.targets))
                    lost_ids = rng2.choice(shots * len(targs), size=n_loss, replace=False)
                    loss_shot, loss_pos = np.divmod(lost_ids, len(targs))
                    loss_pos = targs[loss_pos]
                    new_losses = (~self.current_losses[loss_shot, loss_pos])
                    self.current_losses[loss_shot, loss_pos] = True
                    mask = np.zeros((self.n_qubits, shots), dtype=bool)
                    mask[loss_pos[new_losses], loss_shot[new_losses]] = True
                    fs.do(stim.Circuit(cumulative_circuit_str))
                    cumulative_circuit_str = ''
                    fs.broadcast_pauli_errors(pauli='X', mask=mask, p=0.5)
                    fs.broadcast_pauli_errors(pauli='Z', mask=mask, p=0.5)
            else:
                if op.name in ['M', 'MX', 'MR', 'MRX', 'R', 'RX']:
                    if op.name in ['M', 'MX', 'MR', 'MRX']:
                        measurement_mask.append(self.current_losses[:, op.targets])
                    if op.name in ['MR', 'MRX', 'R', 'RX']:
                        self.current_losses[:, op.targets] = False
                cumulative_circuit_str += str(op) + '\n'
        if len(cumulative_circuit_str) > 0:
            fs.do(stim.Circuit(cumulative_circuit_str))
        return fs, np.concatenate(measurement_mask, axis=1).astype(bool)


    def apply_loss(self, circuit: LogicalCircuit) -> Tuple[List[str], np.ndarray]:
        loss_probs = np.array([op.args for op in circuit if op.name == 'LOSS'])
        n_targs = np.array([len(op.targets) for op in circuit if op.name == 'LOSS'])
        shots = self.batch_size
        rng1, rng2 = self.rng.spawn(2)
        if len(loss_probs) > 0:
            n_losses = iter(rng1.binomial(n_targs * shots, loss_probs))
        else:
            n_losses = iter([])
        lost_qubits = self.current_losses
        ret = [[] for _ in range(shots)]
        measurement_mask = []
        for i, op in enumerate(circuit):
            if op.name == 'LOSS':
                n_loss = next(n_losses)
                if n_loss > 0:
                    targs = np.array(utils.flatten(op.targets))
                    lost_ids = rng2.choice(shots * len(targs), size=n_loss,
                                          replace=False)
                    loss_shot, loss_pos = np.divmod(lost_ids, len(targs))
                    loss_pos = targs[loss_pos]
                    lost_qubits[loss_shot, loss_pos] = True
            else:
                op_str = str(op)
                if op.name in ['M', 'MX', 'MR', 'MRX', 'R', 'RX']:
                    if op.name in ['M', 'MX', 'MR', 'MRX']:
                        measurement_mask.append(lost_qubits[:, op.targets])
                    if op.name in ['MR', 'MRX', 'R', 'RX']:
                        lost_qubits[:, op.targets] = False

                    for i in range(shots):
                        ret[i].append(op_str)
                elif op.name != 'DETECTOR':
                    losses = lost_qubits[:, op.targets]
                    two_qubit_ops = Instruction.TWO_QUBIT_GATES.union(
                        Instruction.TWO_QUBIT_NOISE)
                    if op.name in two_qubit_ops:
                        losses = np.any(losses, axis=-1)
                    any_loss = np.any(losses.reshape((shots, -1)), axis=-1)
                    # saves time: if no loss, no need to remove any qubits
                    for i in range(shots):
                        if not any_loss[i]:
                            ret[i].append(op_str)
                        else:
                            ret[i].append(str(op.remove_qubits(losses[i])))
        circuits = ['\n'.join(x) for x in ret]
        mask = np.concatenate(measurement_mask, axis=1).astype(bool)
        return circuits, mask