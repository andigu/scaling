import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet3D(nn.Module):
    """Unified 3D ResNet implementation with configurable architecture."""
    
    # Classic ResNet configurations [stage1, stage2, stage3, stage4]
    RESNET_CONFIGS = {
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3], 
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3]
    }
    
    def __init__(self, layers=None, embedding_dim=128, architecture='resnet50', 
                 stage3_stride=(1, 1, 1), stage4_stride=(1, 1, 1), use_lstm=False):
        """
        Args:
            layers: List of integers specifying blocks per stage [stage1, stage2, stage3, stage4]
            embedding_dim: Dimension of embeddings
            architecture: String key for predefined architectures ('resnet18', 'resnet34', etc.)
            stage3_stride: 3-tuple (T, H, W) stride for stage 3 downsampling
            stage4_stride: 3-tuple (T, H, W) stride for stage 4 downsampling
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Use provided layers or lookup from architecture
        if layers is None:
            if architecture not in self.RESNET_CONFIGS:
                raise ValueError(f"Unknown architecture '{architecture}'. Available: {list(self.RESNET_CONFIGS.keys())}")
            layers = self.RESNET_CONFIGS[architecture]
        
        self.layers = layers
        self.architecture = architecture
        
        # Embedding layer for detector states (0, 1, 2)
        self.embedding = nn.Embedding(81, embedding_dim)
        
        # Build stages with specified block counts
        self.stage1 = self._make_stage(embedding_dim, embedding_dim * 2, layers[0], dilation=1)
        self.stage2 = self._make_stage(embedding_dim * 2, embedding_dim * 4, layers[1], dilation=1)
        self.stage3 = self._make_stage(embedding_dim * 4, embedding_dim * 8, layers[2], dilation=1, stride=stage3_stride)
        self.stage4 = self._make_stage(embedding_dim * 8, embedding_dim * 16, layers[3], dilation=1, stride=stage4_stride)
        
        # Channel reduction back to embedding_dim
        self.channel_reduce = nn.Conv3d(embedding_dim * 16, embedding_dim, kernel_size=1)

        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        
        # Learnable spatial weighting and projection
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, 2*embedding_dim),
            nn.GELU(),
            nn.Linear(2*embedding_dim, 1)
        )
    
    def _make_stage(self, in_channels, out_channels, num_blocks, dilation=1, stride=(1, 1, 1)):
        blocks = []
        # Determine block type based on architecture
        use_bottleneck = self.architecture in ['resnet50', 'resnet101', 'resnet152']
        
        if use_bottleneck:
            # First block handles channel change and optional downsampling
            blocks.append(self._make_bottleneck_block(in_channels, out_channels, dilation=dilation, downsample=True, stride=stride))
            # Remaining blocks maintain channels with no downsampling
            for _ in range(num_blocks - 1):
                blocks.append(self._make_bottleneck_block(out_channels, out_channels, dilation=dilation))
        else:
            # First block handles channel change and optional downsampling
            blocks.append(self._make_basic_block(in_channels, out_channels, dilation=dilation, downsample=True, stride=stride))
            # Remaining blocks maintain channels with no downsampling
            for _ in range(num_blocks - 1):
                blocks.append(self._make_basic_block(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*blocks)
    
    def _make_bottleneck_block(self, in_channels, out_channels, dilation=1, downsample=False, stride=(1, 1, 1)):
        # ResNet50-style bottleneck: 1x1 -> 3x3 -> 1x1
        # Reduce channels, apply 3x3 conv with dilation, then expand
        mid_channels = out_channels // 4
        
        # Calculate padding to maintain spatial dimensions (adjusted for stride)
        padding = dilation
        
        # Use stride only in the 3x3 conv of the first block if downsampling
        conv_stride = stride if downsample else (1, 1, 1)
        
        layers = nn.Sequential(
            # Pre-activation design
            nn.BatchNorm3d(in_channels),
            nn.GELU(),
            # 1x1 reduction
            nn.Conv3d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm3d(mid_channels),
            nn.GELU(),
            # 3x3 convolution with dilation and optional stride
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=padding, 
                     dilation=dilation, stride=conv_stride),
            nn.BatchNorm3d(mid_channels),
            nn.GELU(),
            # 1x1 expansion
            nn.Conv3d(mid_channels, out_channels, kernel_size=1)
        )
        
        # Skip connection with optional dimension matching
        if downsample or in_channels != out_channels:
            skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=conv_stride)
        else:
            skip = nn.Identity()
            
        return BottleneckBlock(layers, skip)
    
    def _make_basic_block(self, in_channels, out_channels, dilation=1, downsample=False, stride=(1, 1, 1)):
        # ResNet18/34-style basic block: 3x3 -> 3x3
        padding = dilation
        conv_stride = stride if downsample else (1, 1, 1)
        
        layers = nn.Sequential(
            # Pre-activation design
            nn.BatchNorm3d(in_channels),
            nn.GELU(),
            # First 3x3 convolution with optional stride
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding,
                     dilation=dilation, stride=conv_stride),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            # Second 3x3 convolution
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=padding,
                     dilation=dilation)
        )
        
        # Skip connection with optional dimension matching
        if downsample or in_channels != out_channels:
            skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=conv_stride)
        else:
            skip = nn.Identity()
            
        return BottleneckBlock(layers, skip)
    
    def forward(self, x):
        # Input: (batch, time, height, width)
        embedding = self.embedding(x)
        embedding = torch.permute(embedding, (0, 4, 1, 2, 3))  # (batch, embedding_dim, time, height, width)
        
        # Deep ResNet architecture
        embedding = self.stage1(embedding)  # 128 -> 256
        embedding = self.stage2(embedding)  # 256 -> 512
        embedding = self.stage3(embedding)  # 512 -> 1024
        embedding = self.stage4(embedding)  # 1024 -> 2048
        
        # Reduce channels back to embedding_dim
        embedding = self.channel_reduce(embedding)  # 2048 -> 128
        if self.use_lstm:
            embedding = embedding.mean(dim=(-1, -2))  # (batch, embedding_dim, time)
            embedding = embedding.permute(0, 2, 1)  # (batch, time, embedding_dim)
            out, _ = self.lstm(embedding)  # (batch, time, embedding_dim)
            return self.project(out[:,-1,:])
        else:
            return self.project(embedding.mean(dim=(-1,-2,-3)))  # (batch, time, 1)

    @classmethod
    def resnet18(cls, embedding_dim=128, stage3_stride=(1, 1, 1), stage4_stride=(1, 1, 1)):
        """Create ResNet18 variant."""
        return cls(architecture='resnet18', embedding_dim=embedding_dim, 
                  stage3_stride=stage3_stride, stage4_stride=stage4_stride)
    
    @classmethod
    def resnet34(cls, embedding_dim=128, stage3_stride=(1, 1, 1), stage4_stride=(1, 1, 1)):
        """Create ResNet34 variant."""
        return cls(architecture='resnet34', embedding_dim=embedding_dim,
                  stage3_stride=stage3_stride, stage4_stride=stage4_stride)
    
    @classmethod
    def resnet50(cls, embedding_dim=128, stage3_stride=(1, 1, 1), stage4_stride=(1, 1, 1)):
        """Create ResNet50 variant."""
        return cls(architecture='resnet50', embedding_dim=embedding_dim,
                  stage3_stride=stage3_stride, stage4_stride=stage4_stride)
    
    @classmethod
    def resnet101(cls, embedding_dim=128, stage3_stride=(1, 1, 1), stage4_stride=(1, 1, 1)):
        """Create ResNet101 variant."""
        return cls(architecture='resnet101', embedding_dim=embedding_dim,
                  stage3_stride=stage3_stride, stage4_stride=stage4_stride)
    
    @classmethod
    def resnet152(cls, embedding_dim=128, stage3_stride=(1, 1, 1), stage4_stride=(1, 1, 1)):
        """Create ResNet152 variant."""
        return cls(architecture='resnet152', embedding_dim=embedding_dim,
                  stage3_stride=stage3_stride, stage4_stride=stage4_stride)


class BottleneckBlock(nn.Module):
    def __init__(self, main_path, skip_path):
        super().__init__()
        self.main_path = main_path
        self.skip_path = skip_path
    
    def forward(self, x):
        return self.main_path(x) + self.skip_path(x)