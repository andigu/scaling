import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CuGraphRGCNConv
import numpy as np
from torch_geometric import EdgeIndex
from torch.cuda.amp import autocast


class RGCN(nn.Module):
    """RGCN-based model for bivariate bicycle codes, mirroring ResNet3D structure."""
    
    # Same ResNet configurations [stage1, stage2, stage3, stage4]
    RGCN_CONFIGS = {
        'rgcn18': [2, 2, 2, 2],
        'rgcn34': [3, 4, 6, 3], 
        'rgcn50': [3, 4, 6, 3],
        'rgcn101': [3, 4, 23, 3],
        'rgcn152': [3, 8, 36, 3]
    }
    
    def __init__(self, layers=None, embedding_dim=128, architecture='rgcn50', 
                 use_lstm=False, channel_multipliers=None, num_relations=44, num_logical_qubits=1):
        """
        Args:
            layers: List of integers specifying blocks per stage [stage1, stage2, stage3, stage4]
            embedding_dim: Dimension of embeddings
            architecture: String key for predefined architectures ('rgcn18', 'rgcn34', etc.)
            use_lstm: Whether to use LSTM for temporal processing
            channel_multipliers: List of 4 multipliers for channel expansion [stage1, stage2, stage3, stage4]
            num_relations: Number of relation types in the graph
            num_logical_qubits: Number of logical qubits to predict (output dimension)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations
        self.num_logical_qubits = num_logical_qubits
        
        # Use provided layers or lookup from architecture
        if layers is None:
            if architecture not in self.RGCN_CONFIGS:
                raise ValueError(f"Unknown architecture '{architecture}'. Available: {list(self.RGCN_CONFIGS.keys())}")
            layers = self.RGCN_CONFIGS[architecture]
        
        self.layers = layers
        self.architecture = architecture
        
        # Set channel multipliers (default to standard ResNet expansion)
        if channel_multipliers is None:
            channel_multipliers = [2, 4, 8, 16]
        
        if len(channel_multipliers) != 4:
            raise ValueError(f"channel_multipliers must have exactly 4 values, got {len(channel_multipliers)}")
        
        # Round to nearest even number for optimal GPU performance
        def round_to_even(x):
            return max(2, int(2 * round(x / 2)))
        
        # Calculate actual channel counts
        stage_channels = [round_to_even(embedding_dim * m) for m in channel_multipliers]
        self.stage_channels = stage_channels
        self.channel_multipliers = channel_multipliers
        
        # Input embedding layer for detector states (0, 1 detectors)
        self.embedding = nn.Embedding(2, embedding_dim, padding_idx=0)
        
        # Build stages with calculated channel counts
        self.stage1 = self._make_stage(embedding_dim, stage_channels[0], layers[0])
        self.stage2 = self._make_stage(stage_channels[0], stage_channels[1], layers[1])
        self.stage3 = self._make_stage(stage_channels[1], stage_channels[2], layers[2])
        self.stage4 = self._make_stage(stage_channels[2], stage_channels[3], layers[3])
        
        # Channel reduction back to embedding_dim from final stage
        self.channel_reduce = nn.Linear(stage_channels[3], embedding_dim)
        
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        
        # Learnable global pooling and projection
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, 2*embedding_dim),
            nn.GELU(),
            nn.Linear(2*embedding_dim, num_logical_qubits)
        )
    
    def _make_stage(self, in_channels, out_channels, num_blocks):
        blocks = []
        # Determine block type based on architecture
        use_bottleneck = self.architecture in ['rgcn50', 'rgcn101', 'rgcn152']
        
        if use_bottleneck:
            # First block handles channel change
            blocks.append(self._make_bottleneck_block(in_channels, out_channels, first_block=True))
            # Remaining blocks maintain channels
            for _ in range(num_blocks - 1):
                blocks.append(self._make_bottleneck_block(out_channels, out_channels, first_block=False))
        else:
            # First block handles channel change
            blocks.append(self._make_basic_block(in_channels, out_channels, first_block=True))
            # Remaining blocks maintain channels
            for _ in range(num_blocks - 1):
                blocks.append(self._make_basic_block(out_channels, out_channels, first_block=False))
        return nn.ModuleList(blocks)
    
    def _make_bottleneck_block(self, in_channels, out_channels, first_block=False):
        # RGCN-style bottleneck: Linear -> RGCN -> Linear
        mid_channels = out_channels // 4
        
        layers = nn.ModuleDict({
            'norm1': nn.BatchNorm1d(in_channels),
            'linear1': nn.Linear(in_channels, mid_channels),
            'norm2': nn.BatchNorm1d(mid_channels), 
            'rgcn': CuGraphRGCNConv(mid_channels, mid_channels, num_relations=self.num_relations),
            'norm3': nn.BatchNorm1d(mid_channels),
            'linear2': nn.Linear(mid_channels, out_channels)
        })
        
        # Skip connection with optional dimension matching
        if first_block and in_channels != out_channels:
            skip = nn.Linear(in_channels, out_channels)
        else:
            skip = nn.Identity()
            
        return RGCNBlock(layers, skip, use_bottleneck=True)
    
    def _make_basic_block(self, in_channels, out_channels, first_block=False):
        # RGCN-style basic block: RGCN -> RGCN
        
        layers = nn.ModuleDict({
            'norm1': nn.BatchNorm1d(in_channels),
            'rgcn1': CuGraphRGCNConv(in_channels, out_channels, num_relations=self.num_relations),
            'norm2': nn.BatchNorm1d(out_channels),
            'rgcn2': CuGraphRGCNConv(out_channels, out_channels, num_relations=self.num_relations)
        })
        
        # Skip connection with optional dimension matching
        if first_block and in_channels != out_channels:
            skip = nn.Linear(in_channels, out_channels)
        else:
            skip = nn.Identity()
            
        return RGCNBlock(layers, skip, use_bottleneck=False)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (batch_size, num_nodes)
            edge_index: Edge connectivity (2, num_edges)  
            edge_attr: Edge types (num_edges,)
        """
        # Debug logging
        
        # Infer batch dimensions
        batch_size, num_nodes = x.shape
        edge_index = EdgeIndex(edge_index)
        
        # Embed first while keeping batch structure: (batch_size, num_nodes) -> (batch_size, num_nodes, embedding_dim)
        x_flat = x.flatten()  # Temporarily flatten for embedding lookup
        x = self.embedding(x_flat)  # (total_nodes, embedding_dim)
        x = x.reshape(batch_size, num_nodes, -1)  # Reshape back: (batch_size, num_nodes, embedding_dim)
        
        # Pass through RGCN stages with batch-aware processing
        x = self._forward_stage_with_batch(self.stage1, x, edge_index, edge_attr, batch_size, num_nodes)
        x = self._forward_stage_with_batch(self.stage2, x, edge_index, edge_attr, batch_size, num_nodes)  
        x = self._forward_stage_with_batch(self.stage3, x, edge_index, edge_attr, batch_size, num_nodes)
        x = self._forward_stage_with_batch(self.stage4, x, edge_index, edge_attr, batch_size, num_nodes)
        
        # x is now (batch_size, num_nodes, features)
        # Reduce channels back to embedding_dim
        x_flat = x.reshape(-1, x.size(-1))  # Flatten for linear: (batch_size * num_nodes, features)
        x_flat = self.channel_reduce(x_flat)  # (batch_size * num_nodes, embedding_dim)
        x = x_flat.reshape(batch_size, num_nodes, -1)  # Reshape back
        
        if self.use_lstm:
            # For LSTM, we'd need to organize nodes by time
            # This is more complex for graphs - skip for now
            raise NotImplementedError("LSTM not implemented for RGCN yet")
        else:
            # Global mean pooling: (batch_size, num_nodes, features) -> (batch_size, features)
            x = x.mean(dim=1)
            
            return self.project(x)
    
    def _forward_stage(self, stage, x, edge_index, edge_attr):
        """Forward through a stage of RGCN blocks."""
        for block in stage:
            x = block(x, edge_index, edge_attr)
        return x
    
    def _forward_stage_with_batch(self, stage, x, edge_index, edge_attr, batch_size, num_nodes):
        """Forward through a stage with batch-aware processing."""
        # x starts as (batch_size, num_nodes, features)
        for block in stage:
            x = block.forward_with_batch(x, edge_index, edge_attr, batch_size, num_nodes)
        return x
    
    @classmethod
    def rgcn18(cls, embedding_dim=128, channel_multipliers=None, num_relations=12, num_logical_qubits=1):
        """Create RGCN18 variant."""
        return cls(architecture='rgcn18', embedding_dim=embedding_dim, 
                  channel_multipliers=channel_multipliers, num_relations=num_relations,
                  num_logical_qubits=num_logical_qubits)

    @classmethod
    def rgcn34(cls, embedding_dim=128, channel_multipliers=None, num_relations=12, num_logical_qubits=1):
        """Create RGCN34 variant."""
        return cls(architecture='rgcn34', embedding_dim=embedding_dim,
                  channel_multipliers=channel_multipliers, num_relations=num_relations,
                  num_logical_qubits=num_logical_qubits)

    @classmethod
    def rgcn50(cls, embedding_dim=128, channel_multipliers=None, num_relations=12, num_logical_qubits=1):
        """Create RGCN50 variant."""
        return cls(architecture='rgcn50', embedding_dim=embedding_dim,
                  channel_multipliers=channel_multipliers, num_relations=num_relations,
                  num_logical_qubits=num_logical_qubits)
    
    @classmethod
    def rgcn101(cls, embedding_dim=128, channel_multipliers=None, num_relations=12, num_logical_qubits=1):
        """Create RGCN101 variant."""
        return cls(architecture='rgcn101', embedding_dim=embedding_dim,
                  channel_multipliers=channel_multipliers, num_relations=num_relations,
                  num_logical_qubits=num_logical_qubits)

    @classmethod
    def rgcn152(cls, embedding_dim=128, channel_multipliers=None, num_relations=12, num_logical_qubits=1):
        """Create RGCN152 variant."""
        return cls(architecture='rgcn152', embedding_dim=embedding_dim,
                  channel_multipliers=channel_multipliers, num_relations=num_relations,
                  num_logical_qubits=num_logical_qubits)


class RGCNBlock(nn.Module):
    """RGCN block mirroring ResNet bottleneck structure."""
    
    def __init__(self, layers, skip_path, use_bottleneck=False):
        super().__init__()
        self.layers = layers
        self.skip_path = skip_path
        self.use_bottleneck = use_bottleneck
    
    def forward(self, x, edge_index, edge_attr):
        identity = x
        
        if self.use_bottleneck:
            # Bottleneck: norm -> linear -> norm -> rgcn -> norm -> linear
            out = self.layers['norm1'](x)
            out = F.gelu(out)
            out = self.layers['linear1'](out)
            
            out = self.layers['norm2'](out)
            out = F.gelu(out)
            out = self.layers['rgcn'](out, edge_index, edge_attr)
            
            out = self.layers['norm3'](out)
            out = F.gelu(out)
            out = self.layers['linear2'](out)
        else:
            # Basic: norm -> rgcn -> norm -> rgcn
            out = self.layers['norm1'](x)
            out = F.gelu(out)
            out = self.layers['rgcn1'](out, edge_index, edge_attr)
            
            out = self.layers['norm2'](out)
            out = F.gelu(out)
            out = self.layers['rgcn2'](out, edge_index, edge_attr)
        
        # Apply skip connection
        identity = self.skip_path(identity)
        
        return out + identity
    
    def forward_with_batch(self, x, edge_index, edge_attr, batch_size, num_nodes):
        """Forward with batch-aware processing for BatchNorm."""
        # x: (batch_size, num_nodes, features)
        features = x.size(-1)
        identity = x
        if self.use_bottleneck:
            # Bottleneck: norm -> linear -> norm -> rgcn -> norm -> linear
            
            # BatchNorm1d expects (batch, features) or (batch, features, length)
            # Permute to (batch_size, features, num_nodes) for BatchNorm1d
            out = x.permute(0, 2, 1)  # (batch_size, features, num_nodes)
            out = self.layers['norm1'](out)
            out = out.permute(0, 2, 1)  # Back to (batch_size, num_nodes, features)
            out = F.gelu(out)
            
            # Linear layer: flatten, apply, reshape
            out = out.reshape(-1, features)  # (batch_size * num_nodes, features)
            out = self.layers['linear1'](out)
            mid_features = out.size(-1)
            out = out.reshape(batch_size, num_nodes, mid_features)
            
            # Second norm
            out = out.permute(0, 2, 1)
            out = self.layers['norm2'](out)
            out = out.permute(0, 2, 1)
            out = F.gelu(out)
            
            # RGCN expects flattened nodes
            out = out.reshape(-1, mid_features)  # Flatten for RGCN
            # Disable autocast for CuGraphRGCNConv which doesn't support bf16
            with autocast(enabled=False):
                # Convert to fp32 for RGCN
                out_fp32 = out.float()
                edge_attr_fp32 = edge_attr.float() if edge_attr.dtype == torch.bfloat16 else edge_attr
                out = self.layers['rgcn'](out_fp32, edge_index, edge_attr_fp32)
                # Convert back to original dtype if needed
                if out.dtype != out_fp32.dtype:
                    out = out.to(dtype=out_fp32.dtype)
            out = out.reshape(batch_size, num_nodes, -1)  # Reshape back
            
            # Third norm
            out = out.permute(0, 2, 1)
            out = self.layers['norm3'](out)
            out = out.permute(0, 2, 1)
            out = F.gelu(out)
            
            # Final linear
            out = out.reshape(-1, out.size(-1))
            out = self.layers['linear2'](out)
            out = out.reshape(batch_size, num_nodes, -1)
        else:
            # Basic: norm -> rgcn -> norm -> rgcn
            
            # First norm
            out = x.permute(0, 2, 1)  # (batch_size, features, num_nodes)
            out = self.layers['norm1'](out)
            out = out.permute(0, 2, 1)  # Back to (batch_size, num_nodes, features)
            out = F.gelu(out)
            
            # First RGCN
            out = out.reshape(-1, features)  # Flatten for RGCN
            # Disable autocast for CuGraphRGCNConv which doesn't support bf16
            with autocast(enabled=False):
                out_fp32 = out.float()
                edge_attr_fp32 = edge_attr.float() if edge_attr.dtype == torch.bfloat16 else edge_attr
                out = self.layers['rgcn1'](out_fp32, edge_index, edge_attr_fp32)
                if out.dtype != out_fp32.dtype:
                    out = out.to(dtype=out_fp32.dtype)
            new_features = out.size(-1)
            out = out.reshape(batch_size, num_nodes, new_features)  # Reshape back
            
            # Second norm
            out = out.permute(0, 2, 1)
            out = self.layers['norm2'](out)
            out = out.permute(0, 2, 1)
            out = F.gelu(out)
            
            # Second RGCN
            out = out.reshape(-1, new_features)
            # Disable autocast for CuGraphRGCNConv which doesn't support bf16
            with autocast(enabled=False):
                out_fp32 = out.float()
                edge_attr_fp32 = edge_attr.float() if edge_attr.dtype == torch.bfloat16 else edge_attr
                out = self.layers['rgcn2'](out_fp32, edge_index, edge_attr_fp32)
                if out.dtype != out_fp32.dtype:
                    out = out.to(dtype=out_fp32.dtype)
            out = out.reshape(batch_size, num_nodes, -1)
        
        # Apply skip connection
        if not isinstance(self.skip_path, nn.Identity):
            # Need to handle skip path for batched data
            identity_flat = identity.reshape(-1, features)
            identity_flat = self.skip_path(identity_flat)
            identity = identity_flat.reshape(batch_size, num_nodes, -1)
        
        return out + identity