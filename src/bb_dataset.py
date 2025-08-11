from simulate import Algorithm, BivariateBicycle, NoiseModel, CodeFactory, StimSimulator
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
import math


class BivariateBicycleDataset(IterableDataset):
    """
    PyTorch IterableDataset for bivariate bicycle code error correction data.
    
    Generates batches of bivariate bicycle code detector measurements and corresponding 
    logical error labels for training graph neural network decoders.
    
    Supports 3-stage curriculum learning with automatic p adjustment.
    """
    
    def __init__(self, l=6, m=6, rounds_max=9, p=2.0, batch_size=32, 
                 stage_manager=None, num_workers=8, global_step_offset=0):
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
        
        # Cache graph structure for efficiency
        self._graph_structure = None
        self._all_edge_types = None
    
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
    
    def _build_graph_structure(self, rounds):
        """Build the graph structure for the BB code with given rounds."""
        # Create BB code circuit and metadata
        alg = Algorithm.build_memory(cycles=rounds)
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
        
        # Build detector-level graph with temporal connections
        det = pd.DataFrame(metadata.detectors if hasattr(metadata, 'detectors') else [])
        if len(det) == 0:
            # Fallback: create detector metadata from check metadata
            check = pd.DataFrame(metadata.check)
            det = pd.DataFrame({
                'detector_id': np.arange(len(check) * (rounds + 1)),
                'syndrome_id': np.tile(check['check_id'], rounds + 1),
                'time': np.repeat(np.arange(rounds + 1), len(check))
            })
        
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
        noise_model = NoiseModel.get_scaled_noise_model(p).without_loss()
        sim = StimSimulator(alg, noise_model, cf, seed=np.random.randint(0, 2**48))
        
        # Sample syndrome data
        results = sim.sample(shots=batch_size)
        detectors = results.detectors
        logical_errors = results.logical_errors
        
        return detectors, logical_errors, (rounds, p)
    
    def _create_graph_batch(self, detectors, logical_errors, rounds):
        """Convert detector data to PyTorch Geometric batch."""
        if self._graph_structure is None or self._graph_structure.get('rounds') != rounds:
            self._graph_structure = self._build_graph_structure(rounds)
            self._graph_structure['rounds'] = rounds
        
        batch_size, num_detectors = detectors.shape
        
        # Create node features from detector measurements
        # detectors: (batch_size, num_detectors) -> need to reshape for graph
        node_features = detectors.astype(np.float32) + 1  # Convert -1,0,1 to 0,1,2
        
        # Create list of Data objects for batch
        data_list = []
        for i in range(batch_size):
            data = Data(
                x=torch.from_numpy(node_features[i:i+1].T),  # (num_nodes, 1)
                edge_index=torch.from_numpy(self._graph_structure['edge_index']).long(),
                edge_attr=torch.from_numpy(self._graph_structure['edge_attr']).long(),
                y=torch.from_numpy(logical_errors[i].astype(np.float32)),
                num_nodes=self._graph_structure['num_nodes'],
                num_edge_types=self._graph_structure['num_edge_types']
            )
            data_list.append(data)
        
        return data_list
    
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
            detectors, logical_errors, (rounds, p) = self.generate_batch(
                self.l, self.m, self.rounds_max, current_p, self.batch_size
            )
            
            # Convert to graph batch
            graph_batch = self._create_graph_batch(detectors, logical_errors, rounds)
            
            # Yield each graph in the batch
            for graph_data in graph_batch:
                yield graph_data
            
            # Increment local sample count after generating batch
            self.local_sample_count += 1