#!/usr/bin/env python3
"""
Lightning DataModule for Surface Code Error Correction Data.

Handles all data-related concerns including dataset creation, dataloader setup,
worker initialization, and curriculum learning resume support.
"""

import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L
from omegaconf import DictConfig

from dataset import TemporalSurfaceCodeDataset

log = logging.getLogger(__name__)


class CodeDataModule(L.LightningDataModule):
    """
    Lightning DataModule for quantum error correction code datasets.
    
    Handles dataset creation, dataloader setup, worker initialization,
    and curriculum learning resume support in a clean, reusable way.
    """
    
    def __init__(self, cfg: DictConfig, stage_manager=None, batch_size: int = 32, dataset_class=None):
        """
        Args:
            cfg: Hydra configuration object
            stage_manager: Optional StageManager for curriculum learning
            batch_size: Batch size for training
            dataset_class: Dataset class to use (defaults to TemporalSurfaceCodeDataset)
        """
        super().__init__()
        self.cfg = cfg
        self.stage_manager = stage_manager
        self.batch_size = batch_size
        self.dataset_class = dataset_class if dataset_class is not None else TemporalSurfaceCodeDataset
        
        # Dataset will be created in setup()
        self.train_dataset = None
        
        # Global step offset for resume support
        self.global_step_offset = 0
    
    def train_dataloader(self):
        # Create dataset with appropriate parameters based on dataset class
        if self.dataset_class == TemporalSurfaceCodeDataset:
            self.train_dataset = self.dataset_class(
                d=self.cfg.dataset.d,
                rounds_max=self.cfg.dataset.rounds_max,
                p=self.cfg.dataset.p,
                batch_size=self.batch_size,
                mwpm_filter=self.cfg.dataset.mwpm_filter,
                chunking=self.cfg.dataset.get('chunking', (1, 1, 1)),
                stage_manager=self.stage_manager,
                num_workers=self.cfg.hardware.num_workers,
                global_step_offset=self.global_step_offset
            )
        else:
            # For other dataset classes (e.g., BivariateBicycleDataset), use common parameters
            dataset_kwargs = {
                'rounds_max': self.cfg.dataset.rounds_max,
                'p': self.cfg.dataset.p,
                'batch_size': self.batch_size,
                'stage_manager': self.stage_manager,
                'num_workers': self.cfg.hardware.num_workers,
                'global_step_offset': self.global_step_offset
            }
            
            # Add dataset-specific parameters if they exist
            if hasattr(self.cfg.dataset, 'l'):
                dataset_kwargs['l'] = self.cfg.dataset.l
            if hasattr(self.cfg.dataset, 'm'):
                dataset_kwargs['m'] = self.cfg.dataset.m
            if hasattr(self.cfg.dataset, 'd'):
                dataset_kwargs['d'] = self.cfg.dataset.d
            if hasattr(self.cfg.dataset, 'mwpm_filter'):
                dataset_kwargs['mwpm_filter'] = self.cfg.dataset.mwpm_filter
            if hasattr(self.cfg.dataset, 'chunking'):
                dataset_kwargs['chunking'] = self.cfg.dataset.get('chunking', (1, 1, 1))
                
            self.train_dataset = self.dataset_class(**dataset_kwargs)
            
        # Log dataset creation with appropriate parameters
        dataset_info = f"rounds_max={self.cfg.dataset.rounds_max}, p={self.cfg.dataset.p}, batch_size={self.batch_size}"
        if hasattr(self.cfg.dataset, 'd'):
            dataset_info = f"d={self.cfg.dataset.d}, " + dataset_info
        if hasattr(self.cfg.dataset, 'l') and hasattr(self.cfg.dataset, 'm'):
            dataset_info = f"l={self.cfg.dataset.l}, m={self.cfg.dataset.m}, " + dataset_info
        log.info(f"Created dataset ({self.dataset_class.__name__}): {dataset_info}")
        
        if self.stage_manager is not None:
            log.info("Dataset configured for curriculum learning")

        """Create training dataloader with proper worker initialization."""
        if self.train_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        
        # Get local rank for multi-GPU seed diversity
        local_rank = 0
        if hasattr(self.trainer, 'local_rank'):
            local_rank = self.trainer.local_rank
        elif hasattr(self.trainer, 'strategy') and hasattr(self.trainer.strategy, 'local_rank'):
            local_rank = self.trainer.strategy.local_rank
        
        def worker_init_fn(worker_id):
            """Initialize each worker with unique, reproducible seeds."""
            # Calculate global rank for multi-node support
            # global_rank = node_rank * devices_per_node + local_rank
            node_rank = int(os.environ.get('SLURM_NODEID', os.environ.get('NODE_RANK', 0)))
            
            # Get devices per node - use detected num_devices or config
            devices_per_node = 4  # Safe fallback: assume 4 GPUs per node (common case)
            if hasattr(self.trainer, 'num_devices'):
                devices_per_node = self.trainer.num_devices
            elif hasattr(self.trainer, 'strategy') and hasattr(self.trainer.strategy, 'num_processes_per_node'):
                devices_per_node = self.trainer.strategy.num_processes_per_node
            # Try to get from config as backup
            elif hasattr(self.cfg, 'hardware') and hasattr(self.cfg.hardware, 'devices'):
                try:
                    devices_per_node = int(self.cfg.hardware.devices) if self.cfg.hardware.devices != "auto" else 4
                except (ValueError, TypeError):
                    devices_per_node = 4
                
            global_rank = node_rank * devices_per_node + local_rank
            
            # CRITICAL: Use global rank to ensure unique seeds across all nodes and GPUs
            # Formula: base_seed + global_rank * worker_multiplier + worker_id + global_step_offset
            worker_multiplier = 20  # Safety margin for max workers per GPU
            worker_seed = 42 + global_rank * worker_multiplier + worker_id + self.global_step_offset
            
            # Set all random number generators
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            
            log.debug(f"Worker {worker_id} (node {node_rank}, local_rank {local_rank}, global_rank {global_rank}) initialized with seed {worker_seed}")
        
        # Build dataloader kwargs based on num_workers
        dataloader_kwargs = {
            'batch_size': None,  # Dataset handles batching internally
            'shuffle': False,    # Not needed for infinite dataset
            'num_workers': self.cfg.hardware.num_workers,
            'pin_memory': True,
            'worker_init_fn': worker_init_fn
        }
        
        # Only add multiprocessing-specific args if num_workers > 0
        if self.cfg.hardware.num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = self.cfg.hardware.prefetch_factor
            dataloader_kwargs['persistent_workers'] = self.cfg.hardware.persistent_workers
        
        return DataLoader(self.train_dataset, **dataloader_kwargs)
    
    def update_global_step_offset(self, global_step: int):
        """
        Update global step offset for resume support.
        
        This should be called when resuming from a checkpoint to ensure
        the dataset continues curriculum learning from the correct position.
        
        Args:
            global_step: The global step from the checkpoint being resumed
        """
        self.global_step_offset = global_step
        
        # Update the dataset if it's already created
        if self.train_dataset is not None:
            self.train_dataset.global_step_offset = global_step
            log.info(f"Updated DataModule global_step_offset to {global_step} for resume")
        else:
            log.info(f"Set DataModule global_step_offset to {global_step} (will apply when dataset is created)")
    
    def update_batch_size(self, batch_size: int):
        """Update batch size (used during auto-tuning)."""
        self.batch_size = batch_size
        log.info(f"DataModule batch size updated to {batch_size}")
    
    def get_dataset_info(self) -> dict:
        """Get information about the current dataset configuration."""
        info = {
            'dataset_type': self.dataset_class.__name__,
            'rounds_max': self.cfg.dataset.rounds_max,
            'p': self.cfg.dataset.p,
            'batch_size': self.batch_size,
            'num_workers': self.cfg.hardware.num_workers,
            'curriculum_enabled': self.stage_manager is not None,
            'global_step_offset': self.global_step_offset
        }
        
        # Add dataset-specific parameters if they exist
        if hasattr(self.cfg.dataset, 'd'):
            info['d'] = self.cfg.dataset.d
        if hasattr(self.cfg.dataset, 'l'):
            info['l'] = self.cfg.dataset.l
        if hasattr(self.cfg.dataset, 'm'):
            info['m'] = self.cfg.dataset.m
        if hasattr(self.cfg.dataset, 'mwpm_filter'):
            info['mwpm_filter'] = self.cfg.dataset.mwpm_filter
        if hasattr(self.cfg.dataset, 'chunking'):
            info['chunking'] = self.cfg.dataset.get('chunking', (1, 1, 1))
            
        return info


# Backward compatibility alias
SurfaceCodeDataModule = CodeDataModule