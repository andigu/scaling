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
from rgcn import RGCN
from stage_manager import StageManager, CurriculumConfig
from data_module import CodeDataModule
from dataset import TemporalSurfaceCodeDataset
from bb_dataset import BivariateBicycleDataset


class EMA:
    """Exponential Moving Average helper class."""
    def __init__(self, decay=0.999):
        self.decay = decay
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value
        return self.value
    
    def get(self):
        return self.value if self.value is not None else 0.0
    
    def state_dict(self):
        """Return state dictionary for checkpointing."""
        return {
            'decay': self.decay,
            'value': self.value
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.decay = state_dict['decay']
        self.value = state_dict['value']

# Setup logging
log = logging.getLogger(__name__)


def get_dataset_class(cfg: DictConfig):
    """Get the appropriate dataset class based on code type."""
    code_type = cfg.dataset.get('code_type', 'surface_code')
    if code_type == 'bivariate_bicycle':
        return BivariateBicycleDataset
    elif code_type == 'surface_code':
        return TemporalSurfaceCodeDataset
    else:
        raise ValueError(f"Unknown code_type: {code_type}. Must be 'surface_code' or 'bivariate_bicycle'")

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
        self.wandb_run_id_file = experiment_dir / "wandb_run_id.txt"
        
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
    
    def save_wandb_run_id(self, run_id: str):
        """Save W&B run ID for resuming."""
        with open(self.wandb_run_id_file, 'w') as f:
            f.write(run_id)
        log.info(f"Saved W&B run ID: {run_id}")
    
    def load_wandb_run_id(self) -> str:
        """Load W&B run ID for resuming."""
        if self.wandb_run_id_file.exists():
            with open(self.wandb_run_id_file, 'r') as f:
                run_id = f.read().strip()
            log.info(f"Found existing W&B run ID: {run_id}")
            return run_id
        return None
    
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
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        self.lr = cfg.training.lr
        self.weight_decay = cfg.training.weight_decay
        
        # Initialize curriculum learning if enabled
        curriculum_config = CurriculumConfig(
            enabled=cfg.curriculum.enabled,
            stage1_p=cfg.curriculum.stage1_p,
            stage1_steps=cfg.curriculum.stage1_steps,
            stage2_p_end=cfg.curriculum.stage2_p_end,
            stage2_steps=cfg.curriculum.stage2_steps,
            stage3_steps=cfg.curriculum.stage3_steps
        )
        self.stage_manager = StageManager(curriculum_config) if curriculum_config.enabled else None
        
        # Create model based on architecture
        if cfg.model.architecture.startswith('rgcn'):
            # For RGCN, infer num_relations and num_logical_qubits from dataset
            dataset_class = get_dataset_class(cfg)
            if hasattr(dataset_class, '__name__') and 'BivariateBicycle' in dataset_class.__name__:
                # Create a temporary dataset instance to get num_relations and num_logical_qubits
                temp_dataset = dataset_class(
                    l=cfg.dataset.l,
                    m=cfg.dataset.m,
                    rounds_max=cfg.dataset.rounds_max,
                    p=cfg.dataset.p,
                    batch_size=1  # Minimal size for initialization
                )
                num_relations = temp_dataset.get_num_relations()
                num_logical_qubits = temp_dataset.get_num_logical_qubits()
            else:
                num_relations = cfg.model.get('num_relations', 12)  # Fallback
                num_logical_qubits = 1  # Default for surface codes
            
            self.model = RGCN(
                architecture=cfg.model.architecture,
                embedding_dim=cfg.model.embedding_dim,
                channel_multipliers=cfg.model.get('channel_multipliers', None),
                use_lstm=cfg.model.get('use_lstm', False),
                num_relations=num_relations,
                num_logical_qubits=num_logical_qubits
            )
        else:  # ResNet architectures
            self.model = ResNet3D(
                architecture=cfg.model.architecture,
                embedding_dim=cfg.model.embedding_dim,
                channel_multipliers=cfg.model.get('channel_multipliers', None),
                stage3_stride=tuple(cfg.model.stage3_stride),
                stage4_stride=tuple(cfg.model.stage4_stride),
                use_lstm=cfg.model.get('use_lstm', False),
                chunking=cfg.dataset.get('chunking', (1, 1, 1))
            )
        self.reset_metrics()
    
    def setup(self, stage: str) -> None:
        """Called when trainer initializes. Set per-GPU seeds."""
        if stage == "fit":
            # Get local rank (0 for single GPU, 0,1,2,... for multi-GPU)
            local_rank = 0
            if hasattr(self.trainer, 'local_rank'):
                local_rank = self.trainer.local_rank
            elif hasattr(self.trainer, 'strategy') and hasattr(self.trainer.strategy, 'local_rank'):
                local_rank = self.trainer.strategy.local_rank
        
    def reset_metrics(self):
        self.loss_ema = EMA(0.995)
        self.inacc_ema = [EMA(0.995) for _ in range(self.cfg.dataset.rounds_max + 1)]
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # Unified format: (input1, input2, ...), y, (t, p_err)
        x, y, (t, p_err) = batch
        
        # Forward pass with unpacked inputs
        pred = self.model(*x)
        
        loss = F.binary_cross_entropy_with_logits(pred, y)
        
        # Check for stage transitions and log if needed
        stage_prefix = ""
        if self.stage_manager is not None:
            transition = self.stage_manager.check_stage_transition(self.global_step)
            if transition is not None and self.trainer.is_global_zero:
                # Save checkpoint when completing a phase
                old_stage, new_stage = transition
                checkpoint_path = f"{self.trainer.checkpoint_callback.dirpath}/{self.cfg.experiment.name}-phase{old_stage}-step{self.global_step}.ckpt"
                self.trainer.save_checkpoint(checkpoint_path)
                log.info(f"Saved phase {old_stage} completion checkpoint: {checkpoint_path}")
            stage_prefix = self.stage_manager.get_stage_prefix(self.global_step)
            
        # Track metrics (only on main process in multi-GPU)
        if self.trainer.is_global_zero:
            loss_item = loss.item()
            inacc = ((pred > 0) != y).float().mean().item()
            
            self.loss_ema.update(loss_item)
            self.inacc_ema[t-1].update(inacc)
        
        # Log metrics with stage prefix (Lightning handles multi-GPU synchronization)
        self.log(f'{stage_prefix}loss_ema', self.loss_ema.get(), on_step=True, prog_bar=True, sync_dist=False, rank_zero_only=True)
        if batch_idx > 150:
            for t2 in range(len(self.inacc_ema)):
                self.log(f'{stage_prefix}inacc_{t2}', self.inacc_ema[t2].get(), on_step=True, prog_bar=True, sync_dist=False, rank_zero_only=True)
        if self.stage_manager is not None:
            self.log(f'curriculum_p', p_err, on_step=True, prog_bar=True, sync_dist=False, rank_zero_only=True)
        return loss
    
    def configure_optimizers(self):
        if self.cfg.training.optimizer == 'schedulefree':
            optimizer = schedulefree.RAdamScheduleFree(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            optimizer.train()
        elif self.cfg.training.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.training.optimizer}. Must be 'adamw' or 'schedulefree'")
        return optimizer
    
    def state_dict(self):
        """Override to include EMA states and stage manager state in checkpoints."""
        state = super().state_dict()
        
        # Add EMA states
        state['loss_ema'] = self.loss_ema.state_dict()
        state['inacc_ema'] = [ema.state_dict() for ema in self.inacc_ema]
        
        # Add stage manager state for curriculum learning
        if self.stage_manager is not None:
            state['stage_manager'] = self.stage_manager.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Override to restore EMA states and stage manager state from checkpoints."""
        # Extract EMA states and stage manager state before calling super()
        loss_ema_state = state_dict.pop('loss_ema', None)
        inacc_ema_states = state_dict.pop('inacc_ema', None)
        stage_manager_state = state_dict.pop('stage_manager', None)
        
        # Load the rest normally
        result = super().load_state_dict(state_dict, strict)
        
        # Restore EMA states if available
        if loss_ema_state is not None:
            self.loss_ema.load_state_dict(loss_ema_state)
        
        if inacc_ema_states is not None:
            for i, ema_state in enumerate(inacc_ema_states):
                if i < len(self.inacc_ema):
                    self.inacc_ema[i].load_state_dict(ema_state)
        
        # Restore stage manager state if available
        if stage_manager_state is not None and self.stage_manager is not None:
            self.stage_manager.load_state_dict(stage_manager_state)
        
        return result
    
    def on_load_checkpoint(self, checkpoint):
        """Called when loading checkpoint - notify trainer about resume."""
        super().on_load_checkpoint(checkpoint)
        
        # Store the global step for DataModule update
        self._resume_global_step = checkpoint.get('global_step', 0)
        log.info(f"Model loaded from checkpoint at global step {self._resume_global_step}")




def setup_wandb_logger(cfg: DictConfig, experiment_state: ExperimentState, num_devices: int = 1, run_id: str = None):
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
            'channel_multipliers': cfg.model.get('channel_multipliers', [2, 4, 8, 16]),
            'lr': cfg.training.lr,
            'weight_decay': cfg.training.weight_decay,
            'optimizer': cfg.training.optimizer,
            'gradient_clip_val': cfg.training.get('gradient_clip_val', None),
            'gradient_clip_algorithm': cfg.training.get('gradient_clip_algorithm', 'norm'),
            'accumulate_grad_batches': cfg.training.accumulate_grad_batches,
            'num_nodes': cfg.hardware.num_nodes,
            'total_devices': cfg.hardware.num_nodes * num_devices,
            'stage3_stride': cfg.model.stage3_stride,
            'stage4_stride': cfg.model.stage4_stride,
            'dataset_d': cfg.dataset.d,
            'dataset_rounds_max': cfg.dataset.rounds_max,
            'dataset_p': cfg.dataset.p,
            'curriculum_enabled': cfg.curriculum.enabled,
            'curriculum_stage1_p': cfg.curriculum.stage1_p if cfg.curriculum.enabled else None,
            'curriculum_stage1_steps': cfg.curriculum.stage1_steps if cfg.curriculum.enabled else None,
            'curriculum_stage2_p_start': cfg.curriculum.stage1_p if cfg.curriculum.enabled else None,
            'curriculum_stage2_p_end': cfg.curriculum.stage2_p_end if cfg.curriculum.enabled else None,
            'curriculum_stage2_steps': cfg.curriculum.stage2_steps if cfg.curriculum.enabled else None,
            'curriculum_stage3_steps': cfg.curriculum.stage3_steps if cfg.curriculum.enabled else None,
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
    
    
    log.info(f"=== STARTING EXPERIMENT: {cfg.experiment.name} ===")
    log.info(f"PID: {os.getpid()}")
    log.info(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'unknown')}")
    
    # Log multi-node environment variables for debugging
    log.info(f"NODE_RANK: {os.environ.get('NODE_RANK', 'unknown')}")
    log.info(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'unknown')}")
    log.info(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'unknown')}")
    log.info(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'unknown')}")
    log.info(f"SLURM_NODEID: {os.environ.get('SLURM_NODEID', 'unknown')}")
    log.info(f"SLURM_NNODES: {os.environ.get('SLURM_NNODES', 'unknown')}")
    log.info(f"SLURM_NODELIST: {os.environ.get('SLURM_NODELIST', 'unknown')}")
    
    log.info(f"Architecture: {cfg.model.architecture}")
    log.info(f"Embedding Dim: {cfg.model.embedding_dim}")
    log.info(f"Learning Rate: {cfg.training.lr}")
    log.info(f"Weight Decay: {cfg.training.weight_decay}")
    log.info(f"Optimizer: {cfg.training.optimizer}")
    log.info(f"Gradient Clip Val: {cfg.training.get('gradient_clip_val', None)}")
    log.info(f"Gradient Clip Algorithm: {cfg.training.get('gradient_clip_algorithm', 'norm')}")
    log.info(f"Accumulate Grad Batches: {cfg.training.accumulate_grad_batches}")
    log.info(f"Channel Multipliers: {cfg.model.get('channel_multipliers', [2, 4, 8, 16])}")
    
    # Log model-specific parameters
    if cfg.model.architecture.startswith('rgcn'):
        # Get num_relations and num_logical_qubits from dataset for logging
        dataset_class = get_dataset_class(cfg)
        if hasattr(dataset_class, '__name__') and 'BivariateBicycle' in dataset_class.__name__:
            temp_dataset = dataset_class(
                l=cfg.dataset.l,
                m=cfg.dataset.m,
                rounds_max=cfg.dataset.rounds_max,
                p=cfg.dataset.p,
                batch_size=1
            )
            log.info(f"Num Relations: {temp_dataset.get_num_relations()}")
            log.info(f"Num Logical Qubits: {temp_dataset.get_num_logical_qubits()}")
        else:
            log.info(f"Num Relations: {cfg.model.get('num_relations', 12)}")
            log.info(f"Num Logical Qubits: 1")
    else:
        log.info(f"Stage3 Stride: {cfg.model.stage3_stride}")
        log.info(f"Stage4 Stride: {cfg.model.stage4_stride}")
        log.info(f"Use LSTM: {cfg.model.use_lstm}")
        log.info(f"Chunking: {cfg.dataset.chunking}")
    
    # Log dataset-specific parameters
    code_type = cfg.dataset.get('code_type', 'surface_code')
    if code_type == 'bivariate_bicycle':
        log.info(f"Dataset: code_type={code_type}, l={cfg.dataset.l}, m={cfg.dataset.m}, rounds_max={cfg.dataset.rounds_max}, p={cfg.dataset.p}")
    else:
        log.info(f"Dataset: code_type={code_type}, d={cfg.dataset.d}, rounds_max={cfg.dataset.rounds_max}, p={cfg.dataset.p}")
    
    # Log curriculum learning configuration
    if cfg.curriculum.enabled:
        log.info("=== 3-STAGE CURRICULUM LEARNING ENABLED ===")
        log.info(f"Stage 1: p={cfg.curriculum.stage1_p} for {cfg.curriculum.stage1_steps} steps")
        log.info(f"Stage 2: p={cfg.curriculum.stage1_p}→{cfg.curriculum.stage2_p_end} over {cfg.curriculum.stage2_steps} steps")  
        log.info(f"Stage 3: p={cfg.curriculum.stage2_p_end} for {cfg.curriculum.stage3_steps} steps")
        total_curriculum_steps = cfg.curriculum.stage1_steps + cfg.curriculum.stage2_steps + cfg.curriculum.stage3_steps
        log.info(f"Total curriculum steps: {total_curriculum_steps}")
    else:
        log.info("Standard training (no curriculum)")
    
    # Get number of devices for logging
    num_devices = 1
    if cfg.hardware.devices != "auto":
        try:
            num_devices = int(cfg.hardware.devices)
        except (ValueError, TypeError):
            num_devices = 1
    elif cfg.hardware.devices == "auto":
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
    
    # Log multi-node/multi-GPU configuration
    log.info(f"Accelerator: {cfg.hardware.accelerator}")
    log.info(f"Devices per node: {cfg.hardware.devices}")
    log.info(f"Num Nodes: {cfg.hardware.num_nodes}")
    log.info(f"Total Devices: {num_devices * cfg.hardware.num_nodes}")
    log.info(f"Strategy: {cfg.hardware.strategy}")
    log.info(f"Sync BatchNorm: {cfg.hardware.sync_batchnorm}")

    log.info(f"MWPM filtering: {cfg.dataset.mwpm_filter}")
    log.info(f"Batch Size: {cfg.training.batch_size if cfg.training.batch_size else 'auto-tuned'}")
    log.info(f"Precision: {cfg.training.precision}")
    
    # Determine effective max_steps (curriculum overrides if enabled)
    effective_max_steps = cfg.training.max_steps
    if cfg.curriculum.enabled:
        curriculum_total_steps = cfg.curriculum.stage1_steps + cfg.curriculum.stage2_steps + cfg.curriculum.stage3_steps  
        effective_max_steps = curriculum_total_steps
        log.info(f"Max Steps: {effective_max_steps} (curriculum override)")
    else:
        log.info(f"Max Steps: {effective_max_steps}")
        
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
        wandb_logger = setup_wandb_logger(cfg, experiment_state, num_devices, run_id=wandb_run_id)
        
        # If this is a new run, save the W&B run ID for other ranks/restarts
        # Only rank 0 should manage the W&B run ID file
        if wandb_run_id is None and int(os.environ.get('SLURM_PROCID', 0)) == 0:
            # Extract the actual run ID that W&B assigned (only available on rank 0)
            actual_run_id = wandb_logger.experiment.id
            wandb_run_id_file.write_text(actual_run_id)
            wandb_run_id = actual_run_id
            log.info(f"Saved new W&B run ID to file: {actual_run_id}")
        elif wandb_run_id is None:
            log.info("Non-rank-0 process: W&B run ID will be managed by rank 0")
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
        filename=f'{cfg.experiment.name}-{{step}}',
        mode='min',
        train_time_interval=timedelta(minutes=cfg.training.checkpoint_every_minutes),
    )
    
    # num_devices already calculated above for logging
    
    # Handle batch size - manual override or auto-tuning
    batch_size = None
    if cfg.training.batch_size is not None:
        # Manual batch size specified in config
        batch_size = cfg.training.batch_size
        effective_batch_size = batch_size * num_devices * cfg.hardware.num_nodes * cfg.training.accumulate_grad_batches
        log.info(f"Using manual batch size from config: {batch_size}")
        log.info(f"Effective total batch size across {num_devices} device(s) on {cfg.hardware.num_nodes} node(s) with {cfg.training.accumulate_grad_batches} accumulation steps: {effective_batch_size}")
        
        # Save metadata for manual batch size
        experiment_state.save_metadata(
            batch_size=batch_size,
            num_devices=num_devices,
            effective_batch_size=effective_batch_size
        )
    else:
        # Auto-tune batch size with file sharing for multi-GPU
        batch_size_file = experiment_dir / "batch_size.txt"
        
        # Check if batch size file exists (from previous tuning)
        if batch_size_file.exists():
            batch_size = int(batch_size_file.read_text().strip())
            effective_batch_size = batch_size * num_devices * cfg.hardware.num_nodes * cfg.training.accumulate_grad_batches
            log.info(f"Loaded batch size {batch_size} from file")
            log.info(f"Effective total batch size across {num_devices} device(s) on {cfg.hardware.num_nodes} node(s) with {cfg.training.accumulate_grad_batches} accumulation steps: {effective_batch_size}")
            
            # Save metadata for loaded batch size
            experiment_state.save_metadata(
                batch_size=batch_size,
                num_devices=num_devices,
                effective_batch_size=effective_batch_size
            )
        else:
            # Only rank 0 (global rank 0) should do batch size auto-tuning
            is_rank_0 = int(os.environ.get('SLURM_PROCID', 0)) == 0
            
            if is_rank_0:
                # Auto-tune batch size on rank 0 only
                log.info("=== AUTO-TUNING BATCH SIZE (RANK 0) ===")
                tuning_trainer = L.Trainer(
                    accelerator=cfg.hardware.accelerator,
                    devices=1,  # Always use single GPU for batch size tuning
                    precision=cfg.training.precision,
                )
                tuner = Tuner(tuning_trainer)
                dataset_class = get_dataset_class(cfg)
                temp_data_module = CodeDataModule(cfg, stage_manager=model.stage_manager, batch_size=2, dataset_class=dataset_class)
                tuner.scale_batch_size(model, datamodule=temp_data_module, mode="binsearch", steps_per_trial=10)
                log.info("Found optimal batch size " + str(temp_data_module.batch_size))
                model.reset_metrics()
                
                # Use tuned batch size
                batch_size = math.floor(temp_data_module.batch_size * 0.85)
                
                # Save batch size for other ranks
                batch_size_file.write_text(str(batch_size))
                log.info(f"Saved batch size {batch_size} to file")
                
                effective_batch_size = batch_size * num_devices * cfg.hardware.num_nodes * cfg.training.accumulate_grad_batches
                log.info(f"Optimal per-GPU batch size found: {batch_size}")
                log.info(f"Effective total batch size across {num_devices} device(s) on {cfg.hardware.num_nodes} node(s) with {cfg.training.accumulate_grad_batches} accumulation steps: {effective_batch_size}")
                
                # Save metadata
                experiment_state.save_metadata(
                    batch_size=batch_size,
                    num_devices=num_devices,
                    effective_batch_size=effective_batch_size
                )
                
                # Log to W&B if enabled
                if wandb_logger:
                    wandb_logger.log_hyperparams({
                        "auto_tuned_batch_size": batch_size,
                        "num_devices": num_devices,
                        "effective_batch_size": effective_batch_size
                    })
            else:
                # Other ranks wait for batch size file to be created by rank 0
                log.info("=== WAITING FOR BATCH SIZE FROM RANK 0 ===")
                import time
                wait_timeout = 600  # 10 minutes
                wait_start = time.time()
                
                while not batch_size_file.exists():
                    if time.time() - wait_start > wait_timeout:
                        raise RuntimeError(f"Timeout: Batch size file not created by rank 0 after {wait_timeout} seconds")
                    
                    log.info(f"Waiting for batch size file... ({int(time.time() - wait_start)}s elapsed)")
                    time.sleep(10)  # Check every 10 seconds
                
                # Load the batch size that rank 0 found
                batch_size = int(batch_size_file.read_text().strip())
                effective_batch_size = batch_size * num_devices * cfg.hardware.num_nodes * cfg.training.accumulate_grad_batches
                log.info(f"Loaded batch size {batch_size} from rank 0")
                log.info(f"Effective total batch size across {num_devices} device(s) on {cfg.hardware.num_nodes} node(s) with {cfg.training.accumulate_grad_batches} accumulation steps: {effective_batch_size}")
                
                # Save metadata for non-rank-0 processes
                experiment_state.save_metadata(
                    batch_size=batch_size,
                    num_devices=num_devices,
                    effective_batch_size=effective_batch_size
                )
    if wandb_logger:
        wandb_logger.watch(model.model, log="all")
        
        # Log batch size info to wandb (regardless of how it was determined)
        effective_batch_size = batch_size * num_devices * cfg.hardware.num_nodes * cfg.training.accumulate_grad_batches
        wandb_logger.log_hyperparams({
            "final_batch_size": batch_size,
            "num_devices": num_devices,
            "num_nodes": cfg.hardware.num_nodes,
            "accumulate_grad_batches": cfg.training.accumulate_grad_batches,
            "effective_batch_size": effective_batch_size
        })
    
    # Create data module with finalized batch size
    dataset_class = get_dataset_class(cfg)
    data_module = CodeDataModule(cfg, stage_manager=model.stage_manager, batch_size=batch_size, dataset_class=dataset_class)
    
    # Handle resume - update DataModule global step offset if resuming
    if resume_checkpoint:
        # Load the checkpoint to get the global step BEFORE creating the data module
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
        resume_global_step = checkpoint.get('global_step', 0)
        data_module.update_global_step_offset(resume_global_step)
        log.info(f"Updated DataModule global_step_offset to {resume_global_step} for resume")
        
    # Create trainer (this spawns other ranks in multi-GPU mode)
    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        strategy=cfg.hardware.strategy,
        num_nodes=cfg.hardware.num_nodes,
        sync_batchnorm=cfg.hardware.sync_batchnorm,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.get('gradient_clip_val', None),
        gradient_clip_algorithm=cfg.training.get('gradient_clip_algorithm', 'norm'),
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        logger=[csv_logger] + ([wandb_logger] if wandb_logger else []),
        callbacks=[checkpoint_callback],
        max_steps=effective_max_steps,
        log_every_n_steps=cfg.training.log_every_n_steps
    )
        
    # Start training
    log.info("=== STARTING TRAINING ===")
    if resume_checkpoint:
        log.info(f"Resuming from checkpoint: {resume_checkpoint}")
    
    try:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=resume_checkpoint)
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
                batch_size=data_module.batch_size,
                final_checkpoint=final_checkpoint,
                status='completed' if trainer.global_step >= effective_max_steps else 'interrupted'
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
