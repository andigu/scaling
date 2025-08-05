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
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from resnet import ResNet3D
from dataset import TemporalSurfaceCodeDataset, EMA


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')



class ResNet3DTrainer(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.training.lr
        self.batch_size = 2  # Will be auto-tuned
        
        # Create model with configuration
        self.model = ResNet3D(
            architecture=cfg.model.architecture,
            embedding_dim=cfg.model.embedding_dim,
            stage3_stride=tuple(cfg.model.stage3_stride),
            stage4_stride=tuple(cfg.model.stage4_stride)
        )
        self.reset_metrics()
        
    def reset_metrics(self):
        self.loss_ema = EMA(0.995)
        self.inacc_ema = [EMA(0.995) for _ in range(self.cfg.dataset.rounds_max+1)]
    
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train ResNet3D model for quantum error correction."""
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Create model with configuration
    model = ResNet3DTrainer(cfg)

    # Setup logging - both CSV and W&B
    csv_logger = L.pytorch.loggers.CSVLogger(
        save_dir=cfg.experiment.save_dir, 
        name=cfg.experiment.name
    )
    
    # Initialize W&B logger with config
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"{cfg.experiment.name}_{cfg.model.architecture}_embed{cfg.model.embedding_dim}_lr{cfg.training.lr}",
        tags=cfg.wandb.tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir=cfg.experiment.save_dir
    )
    
    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.experiment.checkpoint_dir,
        filename=f'{cfg.experiment.name}-{{step}}-{{loss_ema:.4f}}',
        monitor='loss_ema',
        mode='min',
        every_n_train_steps=cfg.training.checkpoint_every_n_steps,
        save_top_k=-1,
        auto_insert_metric_name=False
    )
    
    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator, 
        precision=cfg.training.precision, 
    )
    
    # Find optimal batch size using binary search
    print("Finding optimal batch size...")
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, mode="binsearch", steps_per_trial=50)
    model.reset_metrics()
    model.batch_size = math.floor(model.batch_size*0.9)
    print(f"Optimal batch size found: {model.batch_size}")
    
    # Log the auto-tuned batch size to W&B
    wandb_logger.log_hyperparams({"auto_tuned_batch_size": model.batch_size})
    
    # Create fresh trainer to avoid state corruption from batch size tuning
    trainer = L.Trainer(
        accelerator=cfg.hardware.accelerator, 
        precision=cfg.training.precision, 
        logger=[csv_logger, wandb_logger],  # Use both loggers
        callbacks=[checkpoint_callback],
        max_steps=cfg.training.max_steps, 
        log_every_n_steps=cfg.training.log_every_n_steps
    )
    
    # Train model with optimized batch size
    trainer.fit(model=model)
    
    # Ensure wandb run is finished
    wandb.finish()


if __name__ == "__main__":
    main()