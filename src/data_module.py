#!/usr/bin/env python3
"""
Lightning DataModule for Surface Code Error Correction Data.

Handles all data-related concerns including dataset creation, dataloader setup,
worker initialization, and curriculum learning resume support.
"""

import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L
from omegaconf import DictConfig

from dataset import TemporalSurfaceCodeDataset

log = logging.getLogger(__name__)


class SurfaceCodeDataModule(L.LightningDataModule):
    """
    Lightning DataModule for Surface Code datasets.
    
    Handles dataset creation, dataloader setup, worker initialization,
    and curriculum learning resume support in a clean, reusable way.
    """
    
    def __init__(self, cfg: DictConfig, stage_manager=None, batch_size: int = 32):
        """
        Args:
            cfg: Hydra configuration object
            stage_manager: Optional StageManager for curriculum learning
            batch_size: Batch size for training
        """
        super().__init__()
        self.cfg = cfg
        self.stage_manager = stage_manager
        self.batch_size = batch_size
        
        # Dataset will be created in setup()
        self.train_dataset = None
        
        # Global step offset for resume support
        self.global_step_offset = 0
    
    def train_dataloader(self):
        self.train_dataset = TemporalSurfaceCodeDataset(
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
            
        log.info(f"Created surface code dataset: d={self.cfg.dataset.d}, "
                f"rounds_max={self.cfg.dataset.rounds_max}, "
                f"p={self.cfg.dataset.p}, batch_size={self.batch_size}")
        
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
            # CRITICAL: Combine base seed (42), local rank, worker_id, and global step offset
            # This ensures unique seeds per worker per GPU while maintaining reproducibility
            worker_seed = 42 + local_rank * 50 + worker_id + self.global_step_offset
            
            # Set all random number generators
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            
            log.debug(f"Worker {worker_id} (rank {local_rank}) initialized with seed {worker_seed}")
        
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Dataset handles batching internally
            shuffle=False,    # Not needed for infinite dataset
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True,
            prefetch_factor=self.cfg.hardware.prefetch_factor,
            persistent_workers=self.cfg.hardware.persistent_workers,
            worker_init_fn=worker_init_fn
        )
    
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
        return {
            'dataset_type': 'TemporalSurfaceCodeDataset',
            'd': self.cfg.dataset.d,
            'rounds_max': self.cfg.dataset.rounds_max,
            'p': self.cfg.dataset.p,
            'batch_size': self.batch_size,
            'mwpm_filter': self.cfg.dataset.mwpm_filter,
            'chunking': self.cfg.dataset.get('chunking', (1, 1, 1)),
            'num_workers': self.cfg.hardware.num_workers,
            'curriculum_enabled': self.stage_manager is not None,
            'global_step_offset': self.global_step_offset
        }