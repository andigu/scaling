#!/usr/bin/env python3
"""
Distance Curriculum Learning Experiment

Tests whether training d=7 then transferring to d=9 improves convergence
compared to training d=9 directly from scratch.

Usage:
    python distance_curriculum.py --mode curriculum  # d=7 -> d=9
    python distance_curriculum.py --mode baseline    # direct d=9
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
import schedulefree
import math
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import time
import logging
from datetime import timedelta
from pathlib import Path
import glob

# Import from existing modules
from resnet import ResNet3D
from data_module import SurfaceCodeDataModule  
from train import EMA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

# Curriculum phase step configuration
PHASE1_STEPS = 15000  # d=7 training steps
PHASE2_STEPS = 50000  # d=9 training steps


class SimpleConfig:
    """Simple config class to replace Hydra."""
    def __init__(self, d=9):
        # Dataset parameters
        self.dataset = SimpleNamespace(
            d=d,
            p=2.1,
            rounds_max=9,
            mwpm_filter=False,
            chunking=(1, 1, 1)
        )
        
        # Model architecture  
        self.model = SimpleNamespace(
            architecture='resnet50',
            embedding_dim=64,
            channel_multipliers=[2, 4, 8, 16],
            stage3_stride=(1, 1, 1),
            stage4_stride=(1, 1, 1),
            use_lstm=False
        )
        
        # Training parameters
        self.training = SimpleNamespace(
            batch_size=None,  # Auto-tune
            lr=3e-4,
            weight_decay=0.0,
            max_steps=PHASE1_STEPS,
            log_every_n_steps=100,
            checkpoint_every_minutes=15,
            precision='bf16-mixed',
            gradient_clip_val=1.0,
            gradient_clip_algorithm='norm',
            accumulate_grad_batches=1
        )
        
        # Hardware settings (single GPU only)
        self.hardware = SimpleNamespace(
            accelerator='auto',
            devices=1,
            strategy='auto',
            num_nodes=1,
            sync_batchnorm=False,
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True
        )
        
        # Experiment settings
        self.experiment = SimpleNamespace(
            name=f'd{d}_experiment',
            base_dir='/n/netscratch/yelin_lab/Everyone/andigu/d-anneal'
        )
        
        # No curriculum learning (we handle it manually)
        self.curriculum = SimpleNamespace(enabled=False)


class SimpleNamespace:
    """Simple namespace for dot notation access."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def get(self, key, default=None):
        return getattr(self, key, default)


class SimpleResNet3DTrainer(L.LightningModule):
    """Simplified trainer without curriculum learning complexity."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.training.lr
        self.weight_decay = cfg.training.weight_decay
        
        # Create model
        self.model = ResNet3D(
            architecture=cfg.model.architecture,
            embedding_dim=cfg.model.embedding_dim,
            channel_multipliers=cfg.model.channel_multipliers,
            stage3_stride=cfg.model.stage3_stride,
            stage4_stride=cfg.model.stage4_stride,
            use_lstm=cfg.model.use_lstm,
            chunking=cfg.dataset.chunking
        )
        
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset EMA tracking metrics."""
        self.loss_ema = EMA(0.995)
        self.inacc_ema = [EMA(0.995) for _ in range(self.cfg.dataset.rounds_max + 1)]
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, (t, p_err) = batch
        pred = self(*x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        
        # Track metrics
        loss_item = loss.item()
        inacc = ((pred > 0) != y).float().mean().item()
        
        self.loss_ema.update(loss_item)
        self.inacc_ema[t-1].update(inacc)
        
        # Log metrics
        self.log('loss_ema', self.loss_ema.get(), on_step=True, prog_bar=True)
        if batch_idx > 150:  # Wait for EMA to stabilize
            for t2 in range(len(self.inacc_ema)):
                self.log(f'inacc_{t2}', self.inacc_ema[t2].get(), on_step=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = schedulefree.RAdamScheduleFree(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer.train()
        return optimizer
    
    def state_dict(self):
        """Include EMA states in checkpoints."""
        state = super().state_dict()
        state['loss_ema'] = self.loss_ema.state_dict()
        state['inacc_ema'] = [ema.state_dict() for ema in self.inacc_ema]
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Restore EMA states from checkpoints.""" 
        loss_ema_state = state_dict.pop('loss_ema', None)
        inacc_ema_states = state_dict.pop('inacc_ema', None)
        
        result = super().load_state_dict(state_dict, strict)
        
        if loss_ema_state is not None:
            self.loss_ema.load_state_dict(loss_ema_state)
        
        if inacc_ema_states is not None:
            for i, ema_state in enumerate(inacc_ema_states):
                if i < len(self.inacc_ema):
                    self.inacc_ema[i].load_state_dict(ema_state)
        
        return result




def extract_step_from_checkpoint(checkpoint_path):
    """Extract step number from checkpoint filename like 'phase1_d7-step=5000.ckpt'."""
    if not checkpoint_path:
        return 0
    
    filename = checkpoint_path.name
    try:
        # Look for pattern like "step=5000" in filename
        if "step=" in filename:
            step_part = filename.split("step=")[1]
            step_str = step_part.split(".")[0]  # Remove .ckpt extension
            return int(step_str)
    except (IndexError, ValueError):
        log.warning(f"Could not extract step from checkpoint: {filename}")
    
    return 0


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in a directory."""
    if not checkpoint_dir.exists():
        return None
    
    # Look for .ckpt files
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    if not ckpt_files:
        return None
    
    # Sort by modification time (most recent first)
    ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return ckpt_files[0]


def get_curriculum_resume_state(experiment_dir):
    """
    Determine curriculum resume strategy based on existing checkpoints.
    
    Returns:
        - ("start_fresh", None): No existing work, start from phase 1
        - ("resume_phase1", checkpoint_path): Resume phase 1 from checkpoint
        - ("start_phase2", None): Phase 1 complete, start phase 2 fresh
        - ("resume_phase2", checkpoint_path): Resume phase 2 from checkpoint  
        - ("completed", None): Both phases complete
    """
    phase1_dir = experiment_dir / "phase1_d7" / "checkpoints"
    phase2_dir = experiment_dir / "phase2_d9" / "checkpoints"
    
    # Check phase 1 state
    phase1_checkpoint = find_latest_checkpoint(phase1_dir)
    phase1_complete = False
    
    if phase1_checkpoint:
        # Check if phase 1 reached required steps by extracting step count
        phase1_steps = extract_step_from_checkpoint(phase1_checkpoint)
        if phase1_steps >= PHASE1_STEPS:
            phase1_complete = True
        # Also check if phase 2 directory exists as indicator phase 1 is done
        elif (experiment_dir / "phase2_d9").exists():
            phase1_complete = True
    
    # Check phase 2 state
    phase2_checkpoint = find_latest_checkpoint(phase2_dir)
    phase2_complete = False
    
    if phase2_checkpoint:
        # Check if phase 2 reached required steps
        phase2_steps = extract_step_from_checkpoint(phase2_checkpoint)
        if phase2_steps >= PHASE2_STEPS:
            phase2_complete = True
    
    # Determine resume strategy
    if phase2_complete:
        return "completed", None
    elif phase2_checkpoint and not phase2_complete:
        return "resume_phase2", phase2_checkpoint
    elif phase1_complete and not phase2_checkpoint:
        return "start_phase2", None
    elif phase1_checkpoint and not phase1_complete:
        return "resume_phase1", phase1_checkpoint
    else:
        return "start_fresh", None


def train_phase(cfg, phase_name, experiment_dir, max_steps, resume_checkpoint=None):
    """Train a single phase (either d=7 or d=9)."""
    
    log.info(f"=== STARTING {phase_name.upper()} ===")
    log.info(f"Distance: d={cfg.dataset.d}")
    log.info(f"Max steps: {max_steps}")
    log.info(f"Learning rate: {cfg.training.lr}")
    log.info(f"Architecture: {cfg.model.architecture}")
    
    phase_dir = experiment_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = SimpleResNet3DTrainer(cfg)
    
    # Auto-tune batch size
    log.info("=== AUTO-TUNING BATCH SIZE ===")
    tuning_trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=1,
        precision=cfg.training.precision,
    )
    tuner = Tuner(tuning_trainer)
    temp_data_module = SurfaceCodeDataModule(cfg, stage_manager=None, batch_size=2)
    tuner.scale_batch_size(model, datamodule=temp_data_module, mode="binsearch", steps_per_trial=10)
    
    batch_size = math.floor(temp_data_module.batch_size * 0.85)
    log.info(f"Optimal batch size: {batch_size}")
    
    # Reset metrics after batch size tuning
    model.reset_metrics()
    
    # Create data module with tuned batch size
    data_module = SurfaceCodeDataModule(cfg, stage_manager=None, batch_size=batch_size)
    
    # Setup CSV logger
    csv_logger = L.pytorch.loggers.CSVLogger(
        save_dir=str(phase_dir),
        name="logs",
        version=""
    )
    
    # Setup checkpointing
    checkpoint_dir = phase_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f'{phase_name}-{{step}}',
        mode='min',
        save_top_k=-1,  # Save all checkpoints
        every_n_train_steps=5000,  # Save every 5k steps
    )
    
    # Create trainer
    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        strategy=cfg.hardware.strategy,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        gradient_clip_algorithm=cfg.training.gradient_clip_algorithm,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
        max_steps=max_steps,
        log_every_n_steps=cfg.training.log_every_n_steps
    )
    
    # Start training
    log.info("=== STARTING TRAINING ===")
    if resume_checkpoint:
        log.info(f"Resuming from: {resume_checkpoint}")
    
    try:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=resume_checkpoint)
        log.info(f"=== {phase_name.upper()} COMPLETED ===")
    except Exception as e:
        log.error(f"=== {phase_name.upper()} FAILED: {e} ===")
        raise
    
    # Return final checkpoint path
    final_checkpoint = checkpoint_callback.last_model_path
    if not final_checkpoint:
        # If no checkpoint was saved, save one now
        final_checkpoint = checkpoint_dir / f"{phase_name}-final.ckpt"
        trainer.save_checkpoint(final_checkpoint)
    
    log.info(f"Final checkpoint: {final_checkpoint}")
    return final_checkpoint, model


def run_curriculum_experiment(base_dir):
    """Run the curriculum experiment: d=7 (50k steps) -> d=9 (50k steps)."""
    
    log.info("=== CURRICULUM EXPERIMENT: d=7 -> d=9 ===")
    experiment_dir = Path(base_dir) / "curriculum_d7_to_d9"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Check resume state
    resume_state, resume_checkpoint = get_curriculum_resume_state(experiment_dir)
    log.info(f"Resume state: {resume_state}")
    
    if resume_checkpoint:
        log.info(f"Resume checkpoint: {resume_checkpoint}")
    
    phase1_checkpoint = None
    phase1_model = None
    
    # Handle Phase 1: d=7 training
    if resume_state == "completed":
        log.info("=== CURRICULUM ALREADY COMPLETED ===")
        # Find the final phase 2 checkpoint
        phase2_dir = experiment_dir / "phase2_d9" / "checkpoints"
        final_checkpoint = find_latest_checkpoint(phase2_dir)
        return final_checkpoint
    
    elif resume_state == "start_fresh":
        log.info("=== STARTING PHASE 1: d=7 TRAINING ===")
        cfg_d7 = SimpleConfig(d=7)
        phase1_checkpoint, phase1_model = train_phase(
            cfg_d7, 
            "phase1_d7", 
            experiment_dir, 
            max_steps=15000
        )
    
    elif resume_state == "resume_phase1":
        log.info("=== RESUMING PHASE 1: d=7 TRAINING ===")
        cfg_d7 = SimpleConfig(d=7)
        phase1_checkpoint, phase1_model = train_phase(
            cfg_d7,
            "phase1_d7", 
            experiment_dir,
            max_steps=15000,
            resume_checkpoint=str(resume_checkpoint)
        )
    
    elif resume_state in ["start_phase2", "resume_phase2"]:
        log.info("=== PHASE 1 ALREADY COMPLETED ===")
    
    # Handle Phase 2: d=9 training
    if resume_state in ["start_fresh", "resume_phase1", "start_phase2"]:
        log.info("=== LOADING d=7 MODEL FOR d=9 TRAINING ===")
        
        # Find phase 1 final checkpoint to use as starting point for phase 2
        phase1_dir = experiment_dir / "phase1_d7" / "checkpoints"  
        phase1_final_checkpoint = find_latest_checkpoint(phase1_dir)
        
        if not phase1_final_checkpoint:
            raise RuntimeError("Phase 1 checkpoint not found - cannot start phase 2")
        
        log.info(f"Using phase 1 checkpoint: {phase1_final_checkpoint}")
        
        # Phase 2: Train d=9 directly from phase 1 checkpoint
        # Since the model architecture is identical, we can load the checkpoint directly
        cfg_d9 = SimpleConfig(d=9)
        log.info("=== STARTING PHASE 2: d=9 TRAINING ===")
        phase2_checkpoint, phase2_model = train_phase(
            cfg_d9,
            "phase2_d9", 
            experiment_dir,
            max_steps=PHASE2_STEPS,
            resume_checkpoint=str(phase1_final_checkpoint)
        )
    
    elif resume_state == "resume_phase2":
        log.info("=== RESUMING PHASE 2: d=9 TRAINING ===")
        cfg_d9 = SimpleConfig(d=9)
        phase2_checkpoint, phase2_model = train_phase(
            cfg_d9,
            "phase2_d9",
            experiment_dir, 
            max_steps=PHASE2_STEPS,
            resume_checkpoint=str(resume_checkpoint)
        )
    
    log.info("=== CURRICULUM EXPERIMENT COMPLETED ===")
    return phase2_checkpoint


def run_baseline_experiment(base_dir):
    """Run the baseline experiment: direct d=9 training for 100k steps."""
    
    log.info("=== BASELINE EXPERIMENT: direct d=9 ===")
    experiment_dir = Path(base_dir) / "baseline_d9"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Train d=9 directly for total curriculum steps
    total_steps = PHASE1_STEPS + PHASE2_STEPS
    log.info(f"Training d=9 directly for {total_steps} steps (equivalent to curriculum total)")
    cfg_d9 = SimpleConfig(d=9)
    final_checkpoint, model = train_phase(
        cfg_d9,
        "baseline_d9",
        experiment_dir, 
        max_steps=total_steps
    )
    
    log.info("=== BASELINE EXPERIMENT COMPLETED ===")
    return final_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Distance Curriculum Learning Experiment")
    parser.add_argument("--mode", choices=["curriculum", "baseline"], required=True,
                        help="Experiment mode: curriculum (d=7->d=9) or baseline (direct d=9)")
    
    args = parser.parse_args()
    
    base_dir = "/n/netscratch/yelin_lab/Everyone/andigu/d-anneal"
    
    if args.mode == "curriculum":
        run_curriculum_experiment(base_dir)
    elif args.mode == "baseline":
        run_baseline_experiment(base_dir)
    
    log.info("=== EXPERIMENT COMPLETED ===")


if __name__ == "__main__":
    main()