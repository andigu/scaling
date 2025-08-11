#!/usr/bin/env python3
"""
Timing comparison between Conv3D and CuGraphRGCNConv.
Compares dense 3D convolution vs sparse graph convolution with equivalent connectivity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CuGraphRGCNConv, CuGraphSAGEConv
from torch_geometric.data import Data
from torch_geometric import EdgeIndex
import numpy as np
import time
from typing import Tuple, List

def create_3d_grid_edges(shape: Tuple[int, int, int], kernel_size: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create edge list for 3D grid graph with kernel_size connectivity.
    Each node connects to all neighbors within kernel radius INCLUDING itself.
    Each edge gets a unique edge type based on relative position.
    
    Args:
        shape: (depth, height, width) of 3D grid
        kernel_size: Size of connectivity kernel (e.g., 3 means 3x3x3 neighborhood)
    
    Returns:
        edge_index: [2, num_edges] tensor of edges
        edge_type: [num_edges] tensor of edge types (0-26 for 3x3x3)
    """
    d, h, w = shape
    radius = kernel_size // 2
    
    edges = []
    edge_types = []
    
    # Create mapping from 3D coordinates to node indices
    def coord_to_idx(z, y, x):
        return z * h * w + y * w + x
    
    # Create mapping from relative position to edge type
    def pos_to_edge_type(dz, dy, dx):
        # Map [-1,0,1] x [-1,0,1] x [-1,0,1] to [0, 26]
        return (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)
    
    # For each node, connect to ALL neighbors within kernel radius INCLUDING itself
    for z in range(d):
        for y in range(h):
            for x in range(w):
                center_idx = coord_to_idx(z, y, x)
                
                # Connect to ALL neighbors within radius (including self)
                for dz in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nz, ny, nx = z + dz, y + dy, x + dx
                            
                            # Check bounds
                            if 0 <= nz < d and 0 <= ny < h and 0 <= nx < w:
                                neighbor_idx = coord_to_idx(nz, ny, nx)
                                edge_type = pos_to_edge_type(dz, dy, dx)
                                
                                edges.append([center_idx, neighbor_idx])
                                edge_types.append(edge_type)
    
    if len(edges) == 0:
        # Fallback: create minimal edge structure
        edges = [[0, 0]]
        edge_types = [0]
    
    edge_index = torch.tensor(edges).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    return edge_index, edge_type

def time_model(model: nn.Module, inputs: tuple, num_warmup: int = 10, 
               num_iterations: int = 100, use_amp: bool = False) -> float:
    """
    Time model execution with CUDA synchronization.
    
    Args:
        model: PyTorch model to time
        inputs: Tuple of input tensors
        num_warmup: Number of warmup iterations
        num_iterations: Number of timing iterations
        use_amp: Whether to use automatic mixed precision (bfloat16)
    
    Returns:
        Average time per iteration in milliseconds
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _ = model(*inputs)
            else:
                _ = model(*inputs)
        torch.cuda.synchronize()
    
    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _ = model(*inputs)
            else:
                _ = model(*inputs)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return np.mean(times)

def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data dimensions
    batch_size = 256
    embedding_dim = 64
    grid_shape = (5, 5, 5)  # depth, height, width
    kernel_size = 3
    
    print(f"Batch size: {batch_size}")
    print(f"Grid shape: {grid_shape}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Kernel size: {kernel_size}")
    print("-" * 50)
    
    # Create Conv3D model
    conv3d_model = nn.Conv3d(
            in_channels=embedding_dim,
            out_channels=embedding_dim, 
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Same padding
            bias=True
        ).to(device)
    
    # Create 3D data for Conv3D [batch, channels, depth, height, width]
    conv3d_input = torch.randn(batch_size, embedding_dim, *grid_shape, 
                              device=device, dtype=torch.float32)
    
    print(f"Conv3D input shape: {conv3d_input.shape}")
    
    # Time Conv3D with bf16 mixed precision
    print("Timing Conv3D with bf16 mixed precision...")
    conv3d_time = time_model(conv3d_model, (conv3d_input,), use_amp=True)
    conv3d_memory = get_memory_usage()
    
    print(f"Conv3D average time: {conv3d_time:.3f} ms")
    print(f"Conv3D memory usage: {conv3d_memory:.1f} MB")
    print()
    
    # Create graph structure for RGCN
    print("Creating 3D grid graph...")
    single_edge_index, single_edge_type = create_3d_grid_edges(grid_shape, kernel_size)
    num_nodes = np.prod(grid_shape)
    num_edges_per_graph = single_edge_index.shape[1]
    
    print(f"Graph nodes per sample: {num_nodes}")
    print(f"Graph edges per sample: {num_edges_per_graph}")
    print(f"Average degree: {num_edges_per_graph / num_nodes:.1f}")
    print(f"Number of edge types: {single_edge_type.max().item() + 1}")
    
    # Batch the graphs by creating offsets for each graph in the batch
    print("Batching graphs...")
    batched_edge_indices = []
    batched_edge_types = []
    
    for batch_idx in range(batch_size):
        # Offset node indices for this graph in the batch
        offset = batch_idx * num_nodes
        offset_edge_index = single_edge_index + offset
        
        batched_edge_indices.append(offset_edge_index)
        batched_edge_types.append(single_edge_type)
    single_edge_index = torch.tensor(single_edge_index, device=device)
    single_edge_type = torch.tensor(single_edge_type, device=device)
    # Concatenate all edge indices and edge types
    batched_edge_index = torch.cat(batched_edge_indices, dim=1).to(device)
    batched_edge_type = torch.cat(batched_edge_types, dim=0).to(device)
    
    total_nodes = batch_size * num_nodes
    total_edges = batch_size * num_edges_per_graph
    
    print(f"Total batched nodes: {total_nodes}")
    print(f"Total batched edges: {total_edges}")
    
    # Create RGCN model with 27 relations (3x3x3 = 27 different relative positions)
    rgcn_model = CuGraphRGCNConv(embedding_dim, embedding_dim, num_relations=27).to(device)
    
    # Create node features for RGCN [batch_size * num_nodes, embedding_dim]
    # CuGraphRGCNConv expects batched node features (flattened across batch dimension)
    rgcn_input = torch.randn(batch_size, num_nodes, embedding_dim, device=device, dtype=torch.float32)
    
    print(f"RGCN input shape: {rgcn_input.shape}")
    print(f"Batched edge index shape: {batched_edge_index.shape}")
    print(f"Batched edge type shape: {batched_edge_type.shape}")
    
    # Time RGCN
    print("Timing CuGraphRGCNConv...")
    rgcn_time = time_model(rgcn_model, (rgcn_input, EdgeIndex(single_edge_index), single_edge_type))
    rgcn_memory = get_memory_usage()
    
    print(f"RGCN average time: {rgcn_time:.3f} ms")
    print(f"RGCN memory usage: {rgcn_memory:.1f} MB")
    print()
    
    # Comparison
    print("=" * 50)
    print("TIMING COMPARISON")
    print("=" * 50)
    print(f"Conv3D:  {conv3d_time:.3f} ms ({conv3d_memory:.1f} MB)")
    print(f"CuGraphRGCNConv:    {rgcn_time:.3f} ms ({rgcn_memory:.1f} MB)")
    print()
    
    # Per-sample timing (divide by batch_size for fair comparison)
    conv3d_per_sample = conv3d_time / batch_size
    rgcn_per_sample = rgcn_time / batch_size
    
    print(f"Per-sample timing:")
    print(f"Conv3D:  {conv3d_per_sample:.4f} ms per sample")
    print(f"CuGraphRGCNConv:    {rgcn_per_sample:.4f} ms per sample")
    print()
    
    if conv3d_time > rgcn_time:
        speedup = conv3d_time / rgcn_time
        print(f"CuGraphRGCNConv is {speedup:.2f}x faster than Conv3D (total batch)")
    else:
        speedup = rgcn_time / conv3d_time
        print(f"Conv3D is {speedup:.2f}x faster than CuGraphRGCNConv (total batch)")
    
    memory_ratio = rgcn_memory / conv3d_memory if conv3d_memory > 0 else float('inf')
    print(f"CuGraphRGCNConv uses {memory_ratio:.2f}x memory compared to Conv3D")
    
    # Parameter count comparison
    conv3d_params = sum(p.numel() for p in conv3d_model.parameters())
    rgcn_params = sum(p.numel() for p in rgcn_model.parameters())
    
    print(f"\nParameter counts:")
    print(f"Conv3D:  {conv3d_params:,} parameters")
    print(f"CuGraphRGCNConv:    {rgcn_params:,} parameters")
    
    if conv3d_params > rgcn_params:
        param_ratio = conv3d_params / rgcn_params
        print(f"Conv3D has {param_ratio:.2f}x more parameters")
    else:
        param_ratio = rgcn_params / conv3d_params
        print(f"CuGraphRGCNConv has {param_ratio:.2f}x more parameters")

if __name__ == "__main__":
    main()