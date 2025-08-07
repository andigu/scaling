#!/usr/bin/env python3
"""
Simplified training script for quantum error correction with native SLURM requeuing.
Each experiment runs independently with directory-based state management.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
import schedulefree
import pymatching
import math
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
import os
import pickle
import time
import logging
from datetime import timedelta
from pathlib import Path
from omegaconf import DictConfig
import hydra
import random

from resnet import ResNet3D
from dataset import TemporalSurfaceCodeDataset, EMA

# Setup logging
log = logging.getLogger(__name__)

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


def set_seed(base_seed: int = 42, local_rank: int = 0):
    """Set seeds for reproducibility with different seeds per GPU.
    
    Args:
        base_seed: Base seed for reproducibility
        local_rank: Local rank of the current process (0 for single GPU)
    """
    # Create different seeds per GPU while maintaining reproducibility
    seed = base_seed + local_rank
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        # Each GPU gets its own seed
        torch.cuda.manual_seed(seed)
    
    log.info(f"Set seed {seed} for local rank {local_rank}")
    
class ExperimentState:
    """Manages experiment state for resuming after preemption."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.metadata_file = experiment_dir / "metadata.pkl"
        self.checkpoints_dir = experiment_dir / "checkpoints"
        self.logs_dir = experiment_dir / "logs"
        self.wandb_dir = experiment_dir / "wandb"
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.wandb_dir.mkdir(exist_ok=True)
    
    def save_metadata(self, **kwargs):
        """Save experiment metadata for resuming."""
        metadata = {
            'timestamp': time.time(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            **kwargs
        }
        
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        log.info(f"Saved metadata: {dict(metadata)}")
    
    def load_metadata(self) -> dict:
        """Load experiment metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            log.info(f"Loaded metadata: {dict(metadata)}")
            return metadata
        return {}
    
    def find_latest_checkpoint(self, experiment_name: str) -> str:
        """Find the most recent checkpoint file."""
        checkpoints = list(self.checkpoints_dir.glob(f"{experiment_name}-*.ckpt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
            log.info(f"Found latest checkpoint: {latest}")
            return str(latest)
        return None
    
    def should_resume(self) -> bool:
        """Check if this experiment should resume from existing state."""
        return self.metadata_file.exists()


class ResNet3DTrainer(L.LightningModule):
    def __init__(self, cfg: DictConfig, batch_size: int = 2):
        super().__init__()
        
        self.batch_size = batch_size
        self.cfg = cfg
        self.lr = cfg.training.lr
        
        # Create model
        self.model = ResNet3D(
            architecture=cfg.model.architecture,
            embedding_dim=cfg.model.embedding_dim,
            stage3_stride=tuple(cfg.model.stage3_stride),
            stage4_stride=tuple(cfg.model.stage4_stride),
            use_lstm=cfg.model.get('use_lstm', False)
        )
        self.reset_metrics()
    
    def setup(self, stage: str) -> None:
        """Called when trainer initializes. Set per-GPU seeds here."""
        if stage == "fit":
            # Get local rank (0 for single GPU, 0,1,2,... for multi-GPU)
            local_rank = 0
            if hasattr(self.trainer, 'local_rank'):
                local_rank = self.trainer.local_rank
            elif hasattr(self.trainer, 'strategy') and hasattr(self.trainer.strategy, 'local_rank'):
                local_rank = self.trainer.strategy.local_rank
            
            # Set different seed for each GPU
            set_seed(base_seed=42, local_rank=local_rank)
        
    def reset_metrics(self):
        self.loss_ema = EMA(0.995)
        self.inacc_ema = [EMA(0.995) for _ in range(self.cfg.dataset.rounds_max + 1)]
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t = x.shape[1]-2
        pred = self(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        
        # Track metrics (only on main process in multi-GPU)
        if self.trainer.is_global_zero:
            loss_item = loss.item()
            inacc = ((pred > 0) != y).float().mean().item()
            
            self.loss_ema.update(loss_item)
            self.inacc_ema[t].update(inacc)
        
        # Log metrics (Lightning handles multi-GPU synchronization)
        self.log('loss_ema', self.loss_ema.get(), on_step=True, prog_bar=True, sync_dist=False, rank_zero_only=True)
        if batch_idx > 150:
            for t2 in range(len(self.inacc_ema)):
                self.log(f'inacc_{t2}', self.inacc_ema[t2].get(), on_step=True, prog_bar=True, sync_dist=False, rank_zero_only=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = schedulefree.RAdamScheduleFree(self.parameters(), lr=self.lr)
        optimizer.train()
        return optimizer
    
    def state_dict(self):
        """Override to include EMA states in checkpoints."""
        state = super().state_dict()
        
        # Add EMA states and batch size
        state['loss_ema'] = self.loss_ema.state_dict()
        state['inacc_ema'] = [ema.state_dict() for ema in self.inacc_ema]
        
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Override to restore EMA states from checkpoints."""
        # Extract EMA states and batch size before calling super()
        loss_ema_state = state_dict.pop('loss_ema', None)
        inacc_ema_states = state_dict.pop('inacc_ema', None)
        
        # Load the rest normally
        result = super().load_state_dict(state_dict, strict)
        
        # Restore EMA states if available
        if loss_ema_state is not None:
            self.loss_ema.load_state_dict(loss_ema_state)
        
        if inacc_ema_states is not None:
            for i, ema_state in enumerate(inacc_ema_states):
                if i < len(self.inacc_ema):
                    self.inacc_ema[i].load_state_dict(ema_state)
        
        return result
    
    def train_dataloader(self):
        # Get local rank for multi-GPU seed diversity
        local_rank = 0
        if hasattr(self.trainer, 'local_rank'):
            local_rank = self.trainer.local_rank
        elif hasattr(self.trainer, 'strategy') and hasattr(self.trainer.strategy, 'local_rank'):
            local_rank = self.trainer.strategy.local_rank
        
        dataset = TemporalSurfaceCodeDataset(
            d=self.cfg.dataset.d,
            rounds_max=self.cfg.dataset.rounds_max,
            p=self.cfg.dataset.p,
            batch_size=self.batch_size,
            mwpm_filter=self.cfg.dataset.mwpm_filter
        )
        
        # Create a worker init function that incorporates local rank
        def worker_init_fn(worker_id):
            # CRITICAL: Combine base seed (42), local rank, and worker_id for unique seeds per worker per GPU
            # This ensures GPU0/Worker0, GPU0/Worker1, GPU1/Worker0, GPU1/Worker1 all have different seeds
            worker_seed = 42 + local_rank * 1000 + worker_id
            
            # Set all random number generators
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            
        return DataLoader(
            dataset, 
            batch_size=None, 
            shuffle=False, 
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True, 
            prefetch_factor=self.cfg.hardware.prefetch_factor,
            persistent_workers=self.cfg.hardware.persistent_workers,
            worker_init_fn=worker_init_fn
        )




def setup_wandb_logger(cfg: DictConfig, experiment_state: ExperimentState, run_id: str = None):
    """Setup W&B logger with optional pre-existing run ID."""
    
    run_name = f"{cfg.experiment.name}_{cfg.model.architecture}_embed{cfg.model.embedding_dim}_lr{cfg.training.lr}"
    
    wandb_kwargs = {
        'project': cfg.wandb.project,
        'name': run_name,
        'tags': cfg.wandb.tags,
        'dir': str(experiment_state.wandb_dir) if experiment_state else "./wandb",
        'config': {
            'experiment_name': cfg.experiment.name,
            'architecture': cfg.model.architecture,
            'embedding_dim': cfg.model.embedding_dim,
            'lr': cfg.training.lr,
            'stage3_stride': cfg.model.stage3_stride,
            'stage4_stride': cfg.model.stage4_stride,
        }
    }
    
    # Add run ID and resume if provided
    if run_id:
        wandb_kwargs['id'] = run_id
        wandb_kwargs['resume'] = 'allow'
        log.info(f"Setting up W&B logger with existing run ID: {run_id}")
    else:
        log.info("Setting up new W&B logger (W&B will assign run ID)")
    
    # Add entity if specified
    if cfg.wandb.entity:
        wandb_kwargs['entity'] = cfg.wandb.entity
    log.info("Initializing wandb logger with params", wandb_kwargs)
    
    wandb_logger = L.pytorch.loggers.WandbLogger(**wandb_kwargs)
    
    return wandb_logger


def train_experiment(cfg: DictConfig):
    """Main training function for a single experiment."""
    
    # Set initial seed for main process (per-GPU seeds will be set in Lightning's setup method)
    set_seed(42, local_rank=0)
    
    log.info(f"=== STARTING EXPERIMENT: {cfg.experiment.name} ===")
    log.info(f"PID: {os.getpid()}")
    log.info(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'unknown')}")
    log.info(f"Architecture: {cfg.model.architecture}")
    log.info(f"Embedding Dim: {cfg.model.embedding_dim}")
    log.info(f"Learning Rate: {cfg.training.lr}")
    log.info(f"Stage3 Stride: {cfg.model.stage3_stride}")
    log.info(f"Stage4 Stride: {cfg.model.stage4_stride}")
    log.info(f"Use LSTM: {cfg.model.use_lstm}")
    log.info(f"Dataset: d={cfg.dataset.d}, rounds_max={cfg.dataset.rounds_max}, p={cfg.dataset.p}")
    
    # Log multi-GPU configuration
    log.info(f"Accelerator: {cfg.hardware.accelerator}")
    log.info(f"Devices: {cfg.hardware.devices}")
    log.info(f"Strategy: {cfg.hardware.strategy}")
    log.info(f"Num Nodes: {cfg.hardware.num_nodes}")
    log.info(f"Sync BatchNorm: {cfg.hardware.sync_batchnorm}")

    log.info(f"MWPM filtering: {cfg.dataset.mwpm_filter}")
    log.info(f"Batch Size: {cfg.training.batch_size if cfg.training.batch_size else 'auto-tuned'}")
    log.info(f"Precision: {cfg.training.precision}")
    log.info(f"Max Steps: {cfg.training.max_steps}")
    log.info(f"Checkpoint every {cfg.training.checkpoint_every_minutes} minutes")
    log.info(f"Log every {cfg.training.log_every_n_steps} steps")
    log.info(f"Num Workers: {cfg.hardware.num_workers}")
    log.info(f"Prefetch Factor: {cfg.hardware.prefetch_factor}")
    log.info(f"Persistent Workers: {cfg.hardware.persistent_workers}")

    # Setup experiment state management
    experiment_dir = Path(cfg.experiment.base_dir) / cfg.experiment.name
    log.info("Experiment directory: " + str(experiment_dir))
    experiment_state = ExperimentState(experiment_dir)
    
    # Find checkpoint for resuming
    resume_checkpoint = experiment_state.find_latest_checkpoint(cfg.experiment.name)
    
    # Setup W&B logger if enabled
    wandb_logger = None
    if cfg.wandb.enabled:
        # Handle W&B run ID sharing across ranks/restarts
        wandb_run_id_file = experiment_dir / "wandb_run_id.txt"
        wandb_run_id = None
        
        # Check if W&B run ID file exists (from previous run or rank 0)
        if wandb_run_id_file.exists():
            wandb_run_id = wandb_run_id_file.read_text().strip()
            log.info(f"Loaded W&B run ID from file: {wandb_run_id}")
        
        # Setup early W&B logger to get run ID
        wandb_logger = setup_wandb_logger(cfg, experiment_state, run_id=wandb_run_id)
        
        # If this is a new run, save the W&B run ID for other ranks/restarts
        if wandb_run_id is None:
            # Extract the actual run ID that W&B assigned
            actual_run_id = wandb_logger.experiment.id
            wandb_run_id_file.write_text(actual_run_id)
            wandb_run_id = actual_run_id
            log.info(f"Saved new W&B run ID to file: {actual_run_id}")
    else:
        log.info("W&B logging disabled in config")
    
    # Create model
    model = ResNet3DTrainer(cfg)
    
    # Setup loggers (Lightning handles multi-GPU automatically)
    # All ranks need the same log directory
    log_dir = str(experiment_dir / "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    csv_logger = L.pytorch.loggers.CSVLogger(
        save_dir=log_dir,
        name="",  # Don't create subdirectory
        version=""  # Don't create version subdirectory
    )
    
    # Setup checkpointing (Lightning handles multi-GPU automatically)
    # All ranks need the same checkpoint directory
    checkpoint_dir = str(experiment_dir / "checkpoints")
    # Ensure checkpoint directory exists on all ranks
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{cfg.experiment.name}-{{step}}-{{loss_ema:.4f}}',
        monitor='loss_ema',
        mode='min',
        train_time_interval=timedelta(minutes=cfg.training.checkpoint_every_minutes),
        save_top_k=3,  # Keep all checkpoints
        auto_insert_metric_name=False
    )
    
    # Get number of devices for logging (needed for both fresh start and resume)
    num_devices = 1
    if cfg.hardware.devices != "auto":
        try:
            num_devices = int(cfg.hardware.devices)
        except (ValueError, TypeError):
            num_devices = 1
    elif cfg.hardware.devices == "auto":
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
    
    # Handle batch size - manual override or auto-tuning
    if cfg.training.batch_size is not None:
        # Manual batch size specified in config
        model.batch_size = cfg.training.batch_size
        log.info(f"Using manual batch size from config: {model.batch_size}")
    else:
        # Auto-tune batch size with file sharing for multi-GPU
        batch_size_file = experiment_dir / "batch_size.txt"
        
        # Check if batch size file exists (from previous tuning)
        if batch_size_file.exists():
            saved_batch_size = int(batch_size_file.read_text().strip())
            model.batch_size = saved_batch_size
            log.info(f"Loaded batch size {model.batch_size} from file")
        elif resume_checkpoint is None:
            # Only tune batch size if not resuming and no saved batch size
            log.info("=== AUTO-TUNING BATCH SIZE ===")
            tuning_trainer = L.Trainer(
                accelerator=cfg.hardware.accelerator,
                devices=1,  # Always use single GPU for batch size tuning
                precision=cfg.training.precision,
            )
            tuner = Tuner(tuning_trainer)
            tuner.scale_batch_size(model, mode="binsearch", steps_per_trial=10)
            log.info("Found optimal batch size " + str(model.batch_size))
            model.batch_size = math.floor(model.batch_size * 0.8)
            model.reset_metrics()
            
            # Save batch size for other ranks
            batch_size_file.write_text(str(model.batch_size))
            log.info(f"Saved batch size {model.batch_size} to file")
            
            effective_batch_size = model.batch_size * num_devices
            log.info(f"Optimal per-GPU batch size found: {model.batch_size}")
            log.info(f"Effective total batch size across {num_devices} device(s): {effective_batch_size}")
            
            # Save metadata
            experiment_state.save_metadata(
                batch_size=model.batch_size,
                num_devices=num_devices,
                effective_batch_size=effective_batch_size
            )
            
            # Log to W&B if enabled
            if wandb_logger:
                wandb_logger.log_hyperparams({
                    "auto_tuned_batch_size": model.batch_size,
                    "num_devices": num_devices,
                    "effective_batch_size": effective_batch_size
                })
        else:
            log.info(f"=== RESUMING FROM CHECKPOINT: {resume_checkpoint} ===")
            effective_batch_size = model.batch_size * num_devices
            if wandb_logger:
                wandb_logger.log_hyperparams({
                    "resumed_batch_size": model.batch_size,
                    "num_devices": num_devices,
                    "effective_batch_size": effective_batch_size
                })
        
    # Create trainer (this spawns other ranks in multi-GPU mode)
    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        strategy=cfg.hardware.strategy,
        num_nodes=cfg.hardware.num_nodes,
        sync_batchnorm=cfg.hardware.sync_batchnorm,
        precision=cfg.training.precision,
        logger=[csv_logger] + ([wandb_logger] if wandb_logger else []),
        callbacks=[checkpoint_callback],
        max_steps=cfg.training.max_steps,
        log_every_n_steps=cfg.training.log_every_n_steps
    )
        
    # Start training
    log.info("=== STARTING TRAINING ===")
    if resume_checkpoint:
        log.info(f"Resuming from checkpoint: {resume_checkpoint}")
    
    try:
        trainer.fit(model=model, ckpt_path=resume_checkpoint)
        log.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
    except KeyboardInterrupt:
        log.info("=== TRAINING INTERRUPTED ===")
    except Exception as e:
        log.error(f"=== TRAINING FAILED: {e} ===")
        raise
    finally:
        # Save final state
        final_checkpoint = checkpoint_callback.last_model_path
        if final_checkpoint:
            experiment_state.save_metadata(
                batch_size=model.batch_size,
                final_checkpoint=final_checkpoint,
                status='completed' if trainer.global_step >= cfg.training.max_steps else 'interrupted'
            )
        
        # Finish W&B run if enabled
        if cfg.wandb.enabled:
            wandb.finish()
    
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Train the experiment
    train_experiment(cfg)


if __name__ == "__main__":
    main()
