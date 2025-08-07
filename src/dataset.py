from simulate import Algorithm, SurfaceCode, NoiseModel, CodeFactory, StimSimulator, Pauli
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset
import math
import pymatching

class EMA:
    """Exponential Moving Average helper class."""
    def __init__(self, decay=0.999):
        self.decay = decay
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value
        return self.value
    
    def get(self):
        return self.value if self.value is not None else 0.0
    
    def state_dict(self):
        """Return state dictionary for checkpointing."""
        return {
            'decay': self.decay,
            'value': self.value
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.decay = state_dict['decay']
        self.value = state_dict['value']


class TemporalSurfaceCodeDataset(IterableDataset):
    """
    PyTorch IterableDataset for temporal surface code error correction data.
    
    Generates batches of surface code detector measurements and corresponding 
    logical error labels for training neural network decoders.
    """
    
    def __init__(self, d=9, rounds_max=9, p=2.0, batch_size=32, mwpm_filter=True):
        """
        Args:
            d: Surface code distance
            rounds_max: Maximum number of error correction rounds
            p: Error probability parameter (scaled noise model parameter)
            batch_size: Batch size for generated data
        """
        super().__init__()
        self.p = p
        self.batch_size = batch_size
        self.d = d
        self.rounds_max = rounds_max
        self.mwpm_filter = mwpm_filter
        
    def __iter__(self):
        """Generate infinite stream of surface code data batches."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id if hasattr(worker_info, 'id') else 'unknown'
            # Use the numpy random state that was already seeded in worker_init_fn
            # This ensures we use the per-GPU, per-worker seed set in train.py
        else: 
            # Main process - set a default seed
            worker_id = 'main'
            np.random.seed(0)
        
        # Use numpy's global random state (which was seeded correctly in worker_init_fn)
        d = self.d
        counter = 0
        
        while True:
            rounds_max = self.rounds_max
            counter += 1
            rounds = np.random.randint(1, rounds_max+1)
            batch_size = math.floor(self.batch_size * rounds_max/rounds)
            
            # Create surface code circuit
            alg = Algorithm.build_memory(cycles=rounds)
            cf = CodeFactory(SurfaceCode, {'d': d})
            noise_model = NoiseModel.get_scaled_noise_model(self.p).without_loss()
            sim = StimSimulator(alg, noise_model, cf, seed=np.random.randint(0, 2**48))
            check_meta = pd.DataFrame(SurfaceCode.create_metadata(d).check)
            results = None
            if self.mwpm_filter:
                results = sim.sample(shots=batch_size * 3) # Oversample to ensure enough valid samples
                stim_circ = sim.stim_circ
                logical_ops = pd.DataFrame(SurfaceCode.create_metadata(d).logical_operators)
                zlog_ids = logical_ops[logical_ops['logical_pauli'] == Pauli.Z]['data_id']
                stim_circ.append_from_stim_program_text("OBSERVABLE_INCLUDE(0) " + " ".join(f"Z{i}" for i in zlog_ids))
                dem = stim_circ.detector_error_model(decompose_errors=True)
                mwpm = pymatching.Matching.from_detector_error_model(dem)
                correct = np.all(mwpm.decode_batch(results.detectors) == results.logical_errors, axis=-1)
                detectors = results.detectors[correct][:batch_size]
                logical_errors = np.ascontiguousarray(results.logical_errors[correct][:batch_size])
            else:
                results = sim.sample(shots=batch_size)
                detectors = results.detectors
                logical_errors = results.logical_errors

            batch_size = len(detectors)
            # Initialize detector array
            det_array = np.zeros((batch_size, rounds+1, d+1, d+1), dtype=np.int32)
            
            # Process detector measurements
            det = pd.DataFrame(results.measurement_tracker.detectors)
            det = det.merge(check_meta, left_on='syndrome_id', right_on='check_id')
            det['time'] -= 1

            # Populate detector array
            time, x, y = det['time'].values, det['pos_x'].values, det['pos_y'].values
            det_array[:, time, x, y] = (detectors[:, det['detector_id']] + 1)
            
            # Reshape and hash for compact representation
            chunked = det_array.reshape((batch_size, rounds+1, (d+1)//2, 2, (d+1)//2, 2)).transpose(0, 1, 2, 4, 3, 5)
            hash_val = np.array([
                [1, 3],
                [9, 27]
            ])[None, None, None, None, ...]
            chunked = np.ascontiguousarray(np.sum(chunked * hash_val, axis=(-1,-2)))

            yield chunked, logical_errors.astype(np.float32)