from simulate import Algorithm, SurfaceCode, NoiseModel, CodeFactory, StimSimulator, Pauli
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset
import math
import pymatching


class TemporalSurfaceCodeDataset(IterableDataset):
    """
    PyTorch IterableDataset for temporal surface code error correction data.
    
    Generates batches of surface code detector measurements and corresponding 
    logical error labels for training neural network decoders.
    
    Supports 3-stage curriculum learning with automatic p adjustment.
    """
    
    def __init__(self, d=9, rounds_max=9, p=2.0, batch_size=32, mwpm_filter=True, chunking=(1,1,1), 
                 stage_manager=None, num_workers=8, global_step_offset=0):
        """
        Args:
            d: Surface code distance
            rounds_max: Maximum number of error correction rounds
            p: Default error probability parameter (used when stage_manager is None)
            batch_size: Batch size for generated data
            mwpm_filter: Enable/disable MWPM filtering for harder samples
            chunking: Chunking parameters for data processing
            stage_manager: StageManager instance for curriculum learning (optional)
            num_workers: Total number of DataLoader workers (for step estimation)
            global_step_offset: Starting global step offset (for resuming)
        """
        super().__init__()
        self.default_p = p
        self.batch_size = batch_size
        self.d = d
        self.rounds_max = rounds_max
        self.mwpm_filter = mwpm_filter
        self.chunking = chunking
        self.stage_manager = stage_manager
        self.num_workers = num_workers
        self.global_step_offset = global_step_offset
        
        # Track local worker sample count (will be set in each worker)
        self.local_sample_count = 0
    
    def _estimate_global_step(self) -> int:
        """Estimate current global step from worker-local sample count."""
        # Each worker generates batches independently
        # Estimated global step = offset + (local_sample_count * num_workers)
        return self.global_step_offset + (self.local_sample_count * self.num_workers)
    
    def get_current_p(self) -> float:
        """Get current p value based on stage manager or default."""
        if self.stage_manager is not None:
            estimated_step = self._estimate_global_step()
            return self.stage_manager.get_current_p(estimated_step)
        else:
            return self.default_p
        
    @staticmethod
    def generate_batch(d, rounds_max, p, batch_size, mwpm_filter=True, chunking=(1,1,1)):
        """Generate a single batch of surface code data."""
        rounds = np.random.randint(1, rounds_max+1)
        batch_size = math.floor(batch_size * rounds_max/rounds)
        
        # Create surface code circuit
        alg = Algorithm.build_memory(cycles=rounds)
        cf = CodeFactory(SurfaceCode, {'d': d})
        noise_model = NoiseModel.get_scaled_noise_model(p).without_loss()
        sim = StimSimulator(alg, noise_model, cf, seed=np.random.randint(0, 2**48))
        check_meta = pd.DataFrame(SurfaceCode.create_metadata(d).check)
        results = None
        if mwpm_filter:
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
        chunk_t, chunk_x, chunk_y = chunking
        n_t, n_x, n_y = math.ceil((rounds+1)/chunk_t), math.ceil((d+1)/chunk_x), math.ceil((d+1)/chunk_y)
        det_array = np.zeros((batch_size, n_t*chunk_t, n_x*chunk_x, n_y*chunk_y), dtype=np.int32)
        
        # Process detector measurements
        det = pd.DataFrame(results.measurement_tracker.detectors)
        det = det.merge(check_meta, left_on='syndrome_id', right_on='check_id')
        det['time'] -= 1

        # Populate detector array
        time, x, y = det['time'].values, det['pos_x'].values, det['pos_y'].values
        det_array[:, time, x, y] = (detectors[:, det['detector_id']] + 1)

        chunked = det_array.reshape((batch_size, n_t, chunk_t, n_x, chunk_x, n_y, chunk_y))
        hash_val = 3**np.arange(chunk_t*chunk_x*chunk_y).reshape((chunk_t, chunk_x, chunk_y))
        chunked = np.ascontiguousarray(np.einsum('btixjyk,ijk->btxy', chunked, hash_val).astype(int))
        return chunked, logical_errors.astype(np.float32), (rounds, p)

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
        
        # Reset local sample count for this worker
        self.local_sample_count = 0
        
        # Use numpy's global random state (which was seeded correctly in worker_init_fn)
        while True:
            # Calculate current p based on estimated global step
            current_p = self.get_current_p()
            
            # Generate batch with current curriculum p value
            yield self.generate_batch(self.d, self.rounds_max, current_p, self.batch_size, self.mwpm_filter, self.chunking)
            
            # Increment local sample count after generating batch
            self.local_sample_count += 1
