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
from datetime import timedelta
from pathlib import Path
from omegaconf import DictConfig
import hydra

from resnet import ResNet3D
from dataset import TemporalSurfaceCodeDataset, EMA

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


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
        
        print(f"Saved metadata: {dict(metadata)}")
    
    def load_metadata(self) -> dict:
        """Load experiment metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            print(f"Loaded metadata: {dict(metadata)}")
            return metadata
        return {}
    
    def find_latest_checkpoint(self, experiment_name: str) -> str:
        """Find the most recent checkpoint file."""
        checkpoints = list(self.checkpoints_dir.glob(f"{experiment_name}-*.ckpt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"Found latest checkpoint: {latest}")
            return str(latest)
        return None
    
    def should_resume(self) -> bool:
        """Check if this experiment should resume from existing state."""
        return self.metadata_file.exists()


class ResNet3DTrainer(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.training.lr
        self.batch_size = 2  # Will be auto-tuned
        
        # Create model
        self.model = ResNet3D(
            architecture=cfg.model.architecture,
            embedding_dim=cfg.model.embedding_dim,
            stage3_stride=tuple(cfg.model.stage3_stride),
            stage4_stride=tuple(cfg.model.stage4_stride)
        )
        self.reset_metrics()
        
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
        
        # Track metrics
        loss_item = loss.item()
        inacc = ((pred > 0) != y).float().mean().item()
        
        self.loss_ema.update(loss_item)
        self.inacc_ema[t].update(inacc)
        self.log('loss_ema', self.loss_ema.get(), on_step=True, prog_bar=True)
        if batch_idx > 150:
            for t2 in range(len(self.inacc_ema)):
                self.log(f'inacc_{t2}', self.inacc_ema[t2].get(), on_step=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = schedulefree.RAdamScheduleFree(self.parameters(), lr=self.lr)
        optimizer.train()
        return optimizer
    
    def state_dict(self):
        """Override to include EMA states in checkpoints."""
        state = super().state_dict()
        
        # Add EMA states
        state['loss_ema'] = self.loss_ema.state_dict()
        state['inacc_ema'] = [ema.state_dict() for ema in self.inacc_ema]
        
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Override to restore EMA states from checkpoints."""
        # Extract EMA states before calling super()
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
        dataset = TemporalSurfaceCodeDataset(
            d=self.cfg.dataset.d,
            rounds_max=self.cfg.dataset.rounds_max,
            p=self.cfg.dataset.p,
            batch_size=self.batch_size
        )
        return DataLoader(
            dataset, 
            batch_size=None, 
            shuffle=False, 
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True, 
            prefetch_factor=self.cfg.hardware.prefetch_factor,
            persistent_workers=self.cfg.hardware.persistent_workers
        )


def setup_wandb_logger(cfg: DictConfig, experiment_state: ExperimentState, resuming: bool = False):
    """Setup W&B logger with resume capability."""
    
    metadata = experiment_state.load_metadata()
    run_name = f"{cfg.experiment.name}_{cfg.model.architecture}_embed{cfg.model.embedding_dim}_lr{cfg.training.lr}"
    
    wandb_kwargs = {
        'project': cfg.wandb.project,
        'name': run_name,
        'tags': cfg.wandb.tags,
        'dir': str(experiment_state.wandb_dir),
        'config': {
            'experiment_name': cfg.experiment.name,
            'architecture': cfg.model.architecture,
            'embedding_dim': cfg.model.embedding_dim,
            'lr': cfg.training.lr,
            'stage3_stride': cfg.model.stage3_stride,
            'stage4_stride': cfg.model.stage4_stride,
            'dataset': cfg.dataset,
        }
    }
    
    # Add entity if specified
    if cfg.wandb.entity:
        wandb_kwargs['entity'] = cfg.wandb.entity
    
    # Try to resume if we have a previous run ID
    if resuming and 'wandb_run_id' in metadata:
        try:
            wandb_kwargs.update({
                'id': metadata['wandb_run_id'],
                'resume': 'allow'  # Use 'allow' instead of 'must' for better fault tolerance
            })
            print(f"Attempting to resume W&B run: {metadata['wandb_run_id']}")
        except Exception as e:
            print(f"Failed to resume W&B run, starting new one: {e}")
            # Remove resume parameters and start fresh
            wandb_kwargs.pop('id', None)
            wandb_kwargs.pop('resume', None)
    
    wandb_logger = L.pytorch.loggers.WandbLogger(**wandb_kwargs)
    
    # Save the W&B run ID for future resuming
    experiment_state.save_metadata(wandb_run_id=wandb_logger.experiment.id)
    
    return wandb_logger


def train_experiment(cfg: DictConfig):
    """Main training function for a single experiment."""
    
    print(f"=== STARTING EXPERIMENT: {cfg.experiment.name} ===")
    print(f"PID: {os.getpid()}")
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'unknown')}")
    print(f"Architecture: {cfg.model.architecture}")
    print(f"Embedding Dim: {cfg.model.embedding_dim}")
    print(f"Learning Rate: {cfg.training.lr}")
    print(f"Stage3 Stride: {cfg.model.stage3_stride}")
    print(f"Stage4 Stride: {cfg.model.stage4_stride}")
    print(f"Dataset: d={cfg.dataset.d}, rounds_max={cfg.dataset.rounds_max}, p={cfg.dataset.p}")
    
    # Setup experiment state management
    experiment_dir = Path(cfg.experiment.save_dir) / cfg.experiment.name
    experiment_state = ExperimentState(experiment_dir)
    
    # Check if we should resume
    resuming = experiment_state.should_resume()
    if resuming:
        print("=== RESUMING EXISTING EXPERIMENT ===")
        metadata = experiment_state.load_metadata()
    else:
        print("=== STARTING NEW EXPERIMENT ===")
        metadata = {}
    
    # Create model
    model = ResNet3DTrainer(cfg)
    
    # Setup loggers
    csv_logger = L.pytorch.loggers.CSVLogger(
        save_dir=str(experiment_state.logs_dir),
        name="",  # Don't create subdirectory
        version=""  # Don't create version subdirectory
    )
    
    wandb_logger = setup_wandb_logger(cfg, experiment_state, resuming)
    
    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(experiment_state.checkpoints_dir),
        filename=f'{cfg.experiment.name}-{{step}}-{{loss_ema:.4f}}',
        monitor='loss_ema',
        mode='min',
        train_time_interval=timedelta(minutes=cfg.training.checkpoint_every_minutes),
        save_top_k=3,  # Keep all checkpoints
        auto_insert_metric_name=False
    )
    
    # Create trainer
    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        precision=cfg.training.precision,
        logger=[csv_logger, wandb_logger],
        callbacks=[checkpoint_callback],
        max_steps=cfg.training.max_steps,
        log_every_n_steps=cfg.training.log_every_n_steps
    )
    
    # Handle batch size tuning and resuming
    resume_checkpoint = None
    if resuming:
        # Try to find existing checkpoint
        resume_checkpoint = experiment_state.find_latest_checkpoint(cfg.experiment.name)
        
        # Restore batch size if available
        if 'batch_size' in metadata:
            model.batch_size = metadata['batch_size']
            print(f"Restored batch size: {model.batch_size}")
        else:
            print("No saved batch size found, will auto-tune")
    
    # Auto-tune batch size only if not resuming or no saved batch size
    if not resuming or 'batch_size' not in metadata:
        print("=== AUTO-TUNING BATCH SIZE ===")
        tuning_trainer = L.Trainer(
            accelerator=cfg.hardware.accelerator,
            precision=cfg.training.precision,
        )
        tuner = Tuner(tuning_trainer)
        tuner.scale_batch_size(model, mode="binsearch", steps_per_trial=50)
        # Only reset metrics if we're not resuming (to preserve restored EMA states)
        if not resuming:
            model.reset_metrics()
        model.batch_size = math.floor(model.batch_size * 0.9)
        print(f"Optimal batch size found: {model.batch_size}")
        
        # Save the tuned batch size
        experiment_state.save_metadata(
            wandb_run_id=wandb_logger.experiment.id,
            batch_size=model.batch_size
        )
        
        # Log to W&B
        wandb_logger.log_hyperparams({"auto_tuned_batch_size": model.batch_size})
        
    # Start training
    print("=== STARTING TRAINING ===")
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    
    try:
        trainer.fit(model=model, ckpt_path=resume_checkpoint)
        print("=== TRAINING COMPLETED SUCCESSFULLY ===")
    except KeyboardInterrupt:
        print("=== TRAINING INTERRUPTED ===")
    except Exception as e:
        print(f"=== TRAINING FAILED: {e} ===")
        raise
    finally:
        # Always save final state
        final_checkpoint = checkpoint_callback.last_model_path
        if final_checkpoint:
            experiment_state.save_metadata(
                wandb_run_id=wandb_logger.experiment.id,
                batch_size=model.batch_size,
                final_checkpoint=final_checkpoint,
                status='completed' if trainer.global_step >= cfg.training.max_steps else 'interrupted'
            )
        
        # Finish W&B run
        wandb.finish()
    
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Train the experiment
    train_experiment(cfg)


if __name__ == "__main__":
    main()