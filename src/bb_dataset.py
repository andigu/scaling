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
    
    def __init__(self, l=6, m=6, rounds_max=9, p=2.0, batch_size=32, 
                 stage_manager=None, num_workers=8, global_step_offset=0, **kwargs):
        """
        Args:
            l: BB code parameter l
            m: BB code parameter m
            rounds_max: Maximum number of error correction rounds
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
        self.rounds_max = rounds_max
        self.stage_manager = stage_manager
        self.num_workers = num_workers
        self.global_step_offset = global_step_offset
        
        # Track local worker sample count (will be set in each worker)
        self.local_sample_count = 0
        
        # Build static code graph structure (independent of rounds)
        self._static_graph_structure = self._build_static_graph_structure()
    
    def get_num_relations(self):
        """Get the total number of relation types (static + temporal)."""
        num_static_edge_types = len(self._static_graph_structure['all_edge_types'])
        return num_static_edge_types * 3  # 3 for temporal connections [-1, 0, 1]
    
    def get_num_logical_qubits(self):
        """Get the number of logical qubits encoded by this BB code."""
        # logical_operators is a numpy structured array with fields: logical_id, logical_pauli, data_id, physical_pauli
        # logical_id goes from 0 to k-1 where k is the number of logical qubits
        logical_ops = self._static_graph_structure['metadata'].logical_operators
        return np.max(logical_ops['logical_id']) + 1
    
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
    
    def _build_static_graph_structure(self):
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
        
        return {
            'graph_df': graph_df,
            'all_edge_types': all_edge_types,
            'metadata': metadata
        }
    
    def _build_spatiotemporal_graph_structure(self, rounds, mt):
        """Build the spatiotemporal graph structure using the static graph and temporal connections.
        
        Args:
            rounds: Number of error correction rounds
            mt: MeasurementTracker from the LogicalCircuit (contains detector metadata)
        """
        # Use pre-computed static structure
        graph_df = self._static_graph_structure['graph_df']
        all_edge_types = self._static_graph_structure['all_edge_types']
        
        # Get actual detector metadata from measurement tracker
        det = pd.DataFrame(mt.detectors)
        
        det_graph = det.merge(graph_df, on='syndrome_id').merge(
            det, left_on='neighbor_syndrome_id', right_on='syndrome_id', suffixes=('', '_nb')
        )
        
        # Filter temporal connections (allow connections within 1 time step)
        det_graph = det_graph[np.abs(det_graph['time'] - det_graph['time_nb']) <= 1]
        
        # Encode temporal information in edge types
        dt = det_graph['time_nb'] - det_graph['time'] + 1  # Maps [-1, 0, 1] to [0, 1, 2]
        det_graph['final_edge_type'] = dt * len(all_edge_types) + det_graph['edge_type']
        
        # Create final edge list
        edge_index = det_graph[['detector_id', 'detector_id_nb']].values.T
        edge_attr = det_graph['final_edge_type'].values
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'num_nodes': len(det),
            'num_edge_types': (len(all_edge_types) * 3),  # 3 for temporal [-1, 0, 1]
            'det_df': det
        }
    
    @staticmethod
    def generate_batch(l, m, rounds_max, p, batch_size):
        """Generate a single batch of bivariate bicycle code data."""
        rounds = np.random.randint(1, rounds_max + 1)
        batch_size = math.floor(batch_size * rounds_max / rounds)
        
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
                self.l, self.m, self.rounds_max, current_p, self.batch_size
            )
            graph_structure = self._build_spatiotemporal_graph_structure(rounds, mt)
            edge_indices, edge_types = graph_structure['edge_index'], graph_structure['edge_attr'] # (2, |E|), (|E|, )
            batch_size, num_nodes = detectors.shape
            edge_indices = (edge_indices[..., None] + np.arange(batch_size)[None, None, :] * num_nodes).reshape((2, -1)).astype(np.int32)
            edge_types = np.repeat(edge_types[:, None], batch_size, axis=1).reshape(-1).astype(np.int32)
            detectors = detectors.astype(np.int32)
            yield (detectors, edge_indices, edge_types), np.squeeze(logical_errors.astype(np.float32), axis=1), (rounds, p)

            # Increment local sample count after generating batch
            self.local_sample_count += 1