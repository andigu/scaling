from simulate import Algorithm, SurfaceCode, NoiseModel, CodeFactory, StimSimulator
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset
import math


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


class TemporalSurfaceCodeDataset(IterableDataset):
    """
    PyTorch IterableDataset for temporal surface code error correction data.
    
    Generates batches of surface code detector measurements and corresponding 
    logical error labels for training neural network decoders.
    """
    
    def __init__(self, d=9, rounds_max=9, p=2.0, batch_size=32):
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
        
    def __iter__(self):
        """Generate infinite stream of surface code data batches."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and hasattr(worker_info, 'seed'): 
            seed = worker_info.seed
        else: 
            seed = 0
        rng = np.random.default_rng(seed)
        d = self.d
        counter = 0
        
        while True:
            rounds_max = self.rounds_max
            counter += 1
            rounds = rng.integers(1, rounds_max+1)
            batch_size = math.floor(self.batch_size * rounds_max/rounds)
            
            # Create surface code circuit
            alg = Algorithm.build_memory(cycles=rounds)
            cf = CodeFactory(SurfaceCode, {'d': d})
            noise_model = NoiseModel.get_scaled_noise_model(self.p).without_loss()
            sim = StimSimulator(alg, noise_model, cf, seed=rng.integers(0, 2**48))
            check_meta = pd.DataFrame(SurfaceCode.create_metadata(d).check)

            # Initialize detector array
            det_array = np.zeros((self.batch_size, rounds+1, d+1, d+1), dtype=np.int32)
            results = sim.sample(shots=self.batch_size)
            
            # Process detector measurements
            det = pd.DataFrame(results.measurement_tracker.detectors)
            det = det.merge(check_meta, left_on='syndrome_id', right_on='check_id')
            det['time'] -= 1

            # Populate detector array
            time, x, y = det['time'].values, det['pos_x'].values, det['pos_y'].values
            det_array[:, time, x, y] = (results.detectors[:, det['detector_id']] + 1)
            
            # Reshape and hash for compact representation
            chunked = det_array.reshape((self.batch_size, rounds+1, (d+1)//2, 2, (d+1)//2, 2)).transpose(0, 1, 2, 4, 3, 5)
            hash = np.array([
                [1, 3],
                [9, 27]
            ])[None, None, None, None, ...]
            chunked = np.ascontiguousarray(np.sum(chunked * hash, axis=(-1,-2)))
            
            yield chunked, results.logical_errors.astype(np.float32)