from simulate import Algorithm, BivariateBicycle, NoiseModel, CodeFactory, StimSimulator, LogicalCircuit
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
import math
from functools import lru_cache


class BivariateBicycleDataset(IterableDataset):
    """
    PyTorch IterableDataset for bivariate bicycle code error correction data.
    
    Generates batches of bivariate bicycle code detector measurements and corresponding 
    logical error labels for training graph neural network decoders.
    
    Supports 3-stage curriculum learning with automatic p adjustment.
    """
    
    def __init__(self, l=6, m=6, rounds=10, p=2.0, batch_size=32, 
                 stage_manager=None, num_workers=8, global_step_offset=0, **kwargs):
        """
        Args:
            l: BB code parameter l
            m: BB code parameter m
            rounds: Number of syndrome rounds
            p: Default error probability parameter (used when stage_manager is None)
            batch_size: Batch size for generated data
            stage_manager: StageManager instance for curriculum learning (optional)
            num_workers: Total number of DataLoader workers (for step estimation)
            global_step_offset: Starting global step offset (for resuming)
        """
        super().__init__()
        self.default_p = p
        self.batch_size = batch_size
        self.l = l
        self.m = m
        self.rounds = rounds
        self.stage_manager = stage_manager
        self.num_workers = num_workers
        self.global_step_offset = global_step_offset
        
        # Track local worker sample count (will be set in each worker)
        self.local_sample_count = 0
        
        # Build static code graph structure (independent of rounds)
        self.graph = self._build_graph()
        meta = BivariateBicycle.create_metadata(l=self.l, m=self.m)
        logical_ops = meta.logical_operators
        self.num_logical_qubits = np.max(logical_ops['logical_id']) + 1
    
    def _estimate_global_step(self) -> int:
        """Estimate current global step from worker-local sample count."""
        return self.global_step_offset + (self.local_sample_count * self.num_workers)
    
    def get_current_p(self) -> float:
        """Get current p value based on stage manager or default."""
        if self.stage_manager is not None:
            estimated_step = self._estimate_global_step()
            return self.stage_manager.get_current_p(estimated_step)
        else:
            return self.default_p
    
    def _build_graph(self):
        """Build the static BB code graph structure (independent of rounds)."""
        # Create BB code metadata
        cf = CodeFactory(BivariateBicycle, {'l': self.l, 'm': self.m})
        metadata = cf().metadata
        
        # Build tanner graph - this creates edges between checks and data
        tanner = pd.DataFrame(metadata.tanner)
        
        # Create check-to-check graph as in bb-test.ipynb
        c2c = tanner.merge(tanner, on='data_id')[['check_id_x', 'check_id_y', 'edge_type_x', 'edge_type_y']]
        all_edge_types = []
        graph_edges = []
        
        for (c1, c2), df in c2c.groupby(['check_id_x', 'check_id_y']):
            edge_types = df[['edge_type_x', 'edge_type_y']].to_numpy()
            edge_types = frozenset([tuple(map(int, x)) for x in edge_types])
            if edge_types not in all_edge_types:
                all_edge_types.append(edge_types)
            c1, c2 = map(int, (c1, c2))
            graph_edges.append((c1, c2, all_edge_types.index(edge_types)))
        
        graph_df = pd.DataFrame(graph_edges, columns=['syndrome_id', 'neighbor_syndrome_id', 'edge_type'])
        graph_df = graph_df.sort_values(by=['syndrome_id', 'edge_type'])

        edge_types = list(sorted(set(map(tuple, graph_df.groupby('syndrome_id').agg(list)['edge_type'].tolist()))))
        # Only two groups of edge types (one runs 0 to 21, other is 22 to 43)
        edge_type_bdr = min(edge_types[1])
        graph_df.loc[graph_df['edge_type'] >= edge_type_bdr, 'edge_type'] -= edge_type_bdr

        graph = np.zeros((graph_df['syndrome_id'].max()+1, edge_type_bdr), dtype=int)
        graph[graph_df['syndrome_id'], graph_df['edge_type']] = graph_df['neighbor_syndrome_id']
        return graph
    
    @staticmethod
    def generate_batch(l, m, rounds, p, batch_size):
        """Generate a single batch of bivariate bicycle code data."""
        
        # Create BB code circuit
        alg = Algorithm.build_memory(cycles=rounds)
        cf = CodeFactory(BivariateBicycle, {'l': l, 'm': m})
        lc, _, mt = LogicalCircuit.from_algorithm(alg, cf)
        noise_model = NoiseModel.get_scaled_noise_model(p).without_loss()
        sim = StimSimulator(alg, noise_model, cf, seed=np.random.randint(0, 2**48))
        
        # Sample syndrome data
        results = sim.sample(shots=batch_size)
        detectors = results.detectors
        logical_errors = results.logical_errors
        
        return detectors, logical_errors, (rounds, p), mt
    
    def __iter__(self):
        """Generate infinite stream of bivariate bicycle code data batches."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id if hasattr(worker_info, 'id') else 'unknown'
        else: 
            worker_id = 'main'
            np.random.seed(0)
        
        # Reset local sample count for this worker
        self.local_sample_count = 0
        
        while True:
            # Calculate current p based on estimated global step
            current_p = self.get_current_p()
            
            # Generate batch with current curriculum p value
            detectors, logical_errors, (rounds, p), mt = self.generate_batch(
                self.l, self.m, self.rounds, current_p, self.batch_size
            )
            synd_id = mt.detectors['syndrome_id']
            detectors = detectors * (synd_id.max()+1) + synd_id[None,:]
            ret = np.zeros((detectors.shape[0], rounds+1, synd_id.max()+1), dtype=np.int32)
            ret[:, mt.detectors['time']-1, mt.detectors['syndrome_id']] = detectors+1

            yield ret, np.squeeze(logical_errors.astype(np.float32), axis=1), (rounds, p)

            # Increment local sample count after generating batch
            self.local_sample_count += 1
    
    def get_num_embeddings(self):
        detectors, logical_errors, (rounds, p), mt = self.generate_batch(
                self.l, self.m, self.rounds, 1.0, self.batch_size
            )
        synd_id = mt.detectors['syndrome_id']
        return 2*(synd_id.max()+1)