import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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
                 use_lstm=False, channel_multipliers=None, static_graph=None, num_logical_qubits=1, num_embeddings=None):
        """
        Args:
            layers: List of integers specifying blocks per stage [stage1, stage2, stage3, stage4]
            embedding_dim: Dimension of embeddings
            architecture: String key for predefined architectures ('rgcn18', 'rgcn34', etc.)
            use_lstm: Whether to use LSTM for temporal processing
            channel_multipliers: List of 4 multipliers for channel expansion [stage1, stage2, stage3, stage4]
            static_graph: Static graph structure (required) - shape (num_nodes, neighborhood_size, 3)
            num_logical_qubits: Number of logical qubits to predict (output dimension)
            num_embeddings: Number of embeddings (mandatory)
        """
        super().__init__()
        
        if static_graph is None:
            raise ValueError("static_graph is required for RGCN")
        
        # Extract neighbors from graph structure: graph[..., 1] contains neighbor indices
        # Derive neighborhood_size from the graph shape
        self.neighborhood_size = static_graph.shape[1]
        self.register_buffer('static_neighbors', torch.tensor(static_graph, dtype=torch.int32))
        
        self.embedding_dim = embedding_dim
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
        
        # Input embedding layer for detector states
        if num_embeddings is None:
            raise ValueError("num_embeddings is required")
        self.embedding = nn.Embedding(num_embeddings+1, embedding_dim)
        
        # Build stages with calculated channel counts
        self.stage1 = self._make_stage(embedding_dim, stage_channels[0], layers[0])
        self.stage2 = self._make_stage(stage_channels[0], stage_channels[1], layers[1])
        self.stage3 = self._make_stage(stage_channels[1], stage_channels[2], layers[2])
        self.stage4 = self._make_stage(stage_channels[2], stage_channels[3], layers[3])
        
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        
        # Learnable global pooling and projection
        self.project = nn.Sequential(
            nn.Linear(stage_channels[3], embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, num_logical_qubits)
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
            # The rgcn convolves over space, but outputs 3x the dimension. 1st dimension is used for previous time
            # second is  used for current time, third is for next time step (essentially break down)
            # conv into space first then time. This is *exact* (no approximation) - but saves mem
            'rgcn': nn.Linear(mid_channels*self.neighborhood_size, 3*mid_channels),
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
            'rgcn1': nn.Linear(in_channels*self.neighborhood_size, 3*out_channels),
            'norm2': nn.BatchNorm1d(out_channels),
            'rgcn2': nn.Linear(out_channels*self.neighborhood_size, 3*out_channels)
        })
        
        # Skip connection with optional dimension matching
        if first_block and in_channels != out_channels:
            skip = nn.Linear(in_channels, out_channels)
        else:
            skip = nn.Identity()
            
        return RGCNBlock(layers, skip, use_bottleneck=False)
    
    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes)
        """
        # Infer batch dimensions
        batch_size, time, num_stabs = x.shape
        # mask = (x > 0).float()
        x = self.embedding(x) # (bs, num_nodes, embed dim)
        
        # Use pre-registered static neighbors
        neighbors = self.static_neighbors
        
        # Pass through RGCN stages with batch-aware processing
        x = self._forward_stage(self.stage1, x, neighbors)
        x = self._forward_stage(self.stage2, x, neighbors)
        x = self._forward_stage(self.stage3, x, neighbors)
        x = self._forward_stage(self.stage4, x, neighbors)

        if self.use_lstm:
            # For LSTM, we'd need to organize nodes by time
            # This is more complex for graphs - skip for now
            raise NotImplementedError("LSTM not implemented for RGCN yet")
        else:
            # Global mean pooling: (batch_size, num_nodes, features) -> (batch_size, features)
            x = x.mean(dim=(-2, -3))
            return self.project(x)

    def _forward_stage(self, stage, x, neighbors):
        """Forward through a stage of RGCN blocks."""
        for block in stage:
            x = block(x, neighbors)
        return x
    
    @classmethod
    def rgcn18(cls, embedding_dim=128, channel_multipliers=None, static_graph=None, num_logical_qubits=1, num_embeddings=None):
        """Create RGCN18 variant."""
        return cls(architecture='rgcn18', embedding_dim=embedding_dim, 
                  channel_multipliers=channel_multipliers, static_graph=static_graph,
                  num_logical_qubits=num_logical_qubits, num_embeddings=num_embeddings)

    @classmethod
    def rgcn34(cls, embedding_dim=128, channel_multipliers=None, static_graph=None, num_logical_qubits=1, num_embeddings=None):
        """Create RGCN34 variant."""
        return cls(architecture='rgcn34', embedding_dim=embedding_dim,
                  channel_multipliers=channel_multipliers, static_graph=static_graph,
                  num_logical_qubits=num_logical_qubits, num_embeddings=num_embeddings)

    @classmethod
    def rgcn50(cls, embedding_dim=128, channel_multipliers=None, static_graph=None, num_logical_qubits=1, num_embeddings=None):
        """Create RGCN50 variant."""
        return cls(architecture='rgcn50', embedding_dim=embedding_dim,
                  channel_multipliers=channel_multipliers, static_graph=static_graph,
                  num_logical_qubits=num_logical_qubits, num_embeddings=num_embeddings)
    
    @classmethod
    def rgcn101(cls, embedding_dim=128, channel_multipliers=None, static_graph=None, num_logical_qubits=1, num_embeddings=None):
        """Create RGCN101 variant."""
        return cls(architecture='rgcn101', embedding_dim=embedding_dim,
                  channel_multipliers=channel_multipliers, static_graph=static_graph,
                  num_logical_qubits=num_logical_qubits, num_embeddings=num_embeddings)

    @classmethod
    def rgcn152(cls, embedding_dim=128, channel_multipliers=None, static_graph=None, num_logical_qubits=1, num_embeddings=None):
        """Create RGCN152 variant."""
        return cls(architecture='rgcn152', embedding_dim=embedding_dim,
                  channel_multipliers=channel_multipliers, static_graph=static_graph,
                  num_logical_qubits=num_logical_qubits, num_embeddings=num_embeddings)


class RGCNBlock(nn.Module):
    """RGCN block mirroring ResNet bottleneck structure."""
    
    def __init__(self, layers, skip_path, use_bottleneck=False):
        super().__init__()
        self.layers = layers
        self.skip_path = skip_path
        self.use_bottleneck = use_bottleneck
    
    def forward(self, x, neighbors):
        """Forward with batch-aware processing for BatchNorm.
        
        Args:
            x: Node features (batch_size, num_nodes, features)
            neighbors: Pre-extracted neighbor indices from static graph
        """
        # x: (batch_size, num_nodes, features)
        identity = x
        batch_size, time, num_stabs, features = x.shape
        if self.use_bottleneck:
            # Bottleneck: norm -> linear -> norm -> rgcn -> norm -> linear
            
            # BatchNorm1d expects (batch, features) or (batch, features, length)
            # Permute to (batch_size, features, num_nodes) for BatchNorm1d
            out = self.layers['norm1'](x.flatten(1,2).permute(0, 2, 1)).permute(0, 2, 1).view_as(x)
            out = F.gelu(out)
            
            out = self.layers['linear1'](out)
            
            # Second norm
            out = self.layers['norm2'](out.flatten(1,2).permute(0, 2, 1)).permute(0, 2, 1).view_as(out)
            out = F.gelu(out)

            x0, x1, x2 = self.layers['rgcn'](out[:, :, neighbors].flatten(start_dim=-2)).chunk(3, dim=-1)
            out = ((F.pad(x0[:,:-1], (0,0,0,0,1,0,0,0)) + x1 + F.pad(x2[:,1:], (0,0,0,0,0,1,0,0)))/math.sqrt(3))

            out = self.layers['norm3'](out.flatten(1,2).permute(0, 2, 1)).permute(0, 2, 1).view_as(out)
            out = F.gelu(out)
            
            out = self.layers['linear2'](out)
        else:
            # Basic: norm -> rgcn -> norm -> rgcn
            
            # First norm
            out = self.layers['norm1'](x.flatten(1,2).permute(0, 2, 1)).permute(0, 2, 1).view_as(x)
            out = F.gelu(out)
            
            # First RGCN
            x0, x1, x2 = self.layers['rgcn1'](out[:, :, neighbors].flatten(start_dim=-2)).chunk(3, dim=-1)
            out = ((F.pad(x0[:,:-1], (0,0,0,0,1,0,0,0)) + x1 + F.pad(x2[:,1:], (0,0,0,0,0,1,0,0)))/math.sqrt(3))

            # Second norm
            out = self.layers['norm2'](out.flatten(1,2).permute(0, 2, 1)).permute(0, 2, 1).view_as(out)
            out = F.gelu(out)
            
            # Second RGCN
            x0, x1, x2 = self.layers['rgcn2'](out[:, :, neighbors].flatten(start_dim=-2)).chunk(3, dim=-1)
            out = (F.pad(x0[:,:-1], (0,0,0,0,1,0,0,0)) + x1 + F.pad(x2[:,1:], (0,0,0,0,0,1,0,0)))/math.sqrt(3)
        
        return out + self.skip_path(identity)