import stim
import numpy as np
import pymatching
import sinter
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import lightning as L
import wandb
import math
import random

def get_circ(n, rounds, p):
    circ = stim.Circuit("R " + " ".join(map(str, range(n))))
    circ.append_from_stim_program_text(f"X_ERROR({p}) " + " ".join(map(str, range(n))))
    for _ in range(rounds):
        circ.append_from_stim_program_text("R " + " ".join(map(str, range(n, 2*n))))
        circ.append_from_stim_program_text(f"X_ERROR({p}) " + " ".join(map(str, range(n, 2*n))))
        pairs = np.array([(i, i + n) for i in range(n-1)]).flatten()
        circ.append_from_stim_program_text("CX " + " ".join(map(str, pairs)))
        circ.append_from_stim_program_text(f"DEPOLARIZE2({p}) " + " ".join(map(str, pairs)))

        pairs = np.array([(i+1, i + n) for i in range(n-1)]).flatten()
        circ.append_from_stim_program_text("CX " + " ".join(map(str, pairs)))
        circ.append_from_stim_program_text(f"DEPOLARIZE2({p}) " + " ".join(map(str, pairs)))

        circ.append_from_stim_program_text("M " + " ".join(map(str, range(n, 2*n))))
        for i in range(1,n):
            circ.append_from_stim_program_text(f"DETECTOR rec[{-i}]")
    
    circ.append_from_stim_program_text(f"X_ERROR({p}) " + " ".join(map(str, range(n))))
    circ.append_from_stim_program_text("M " + " ".join(map(str, range(n))))
    for i in range(1, n):
        circ.append_from_stim_program_text(f"DETECTOR rec[{-i}] rec[{-i-1}]")
    circ.append_from_stim_program_text("OBSERVABLE_INCLUDE(0) " + " ".join(f"Z{i}" for i in range(n)))
    return circ

def generate_tasks():
    rounds = 20
    for n in [5, 7, 9, 11, 13]:
        # for rounds in [5, 15, 25, 35]:
        for p in np.logspace(-2, -0.5, 10):
            yield sinter.Task(
            circuit=get_circ(n, rounds, p),
            json_metadata={
                'n': n,
                'p': p,
                'rounds': rounds,
            },
        )

def collect_matching():
    # Collect the samples (takes a few minutes).
    samples = sinter.collect(
        num_workers=32,
        max_shots=1_000_000,
        tasks=generate_tasks(),
        decoders=['pymatching'],
        print_progress=True,
    )

    # Print samples as CSV data.
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())

    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        group_func=lambda stat: f"n={stat.json_metadata['n']}",
        x_func=lambda stat: stat.json_metadata['p'],
    )
    ax.loglog()
    ax.legend()
    fig.savefig('repitition.png')



class Decoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(2, embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding='same'),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding='same'),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding='same'),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding='same'),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
        )
        self.proj = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, syndromes):
        X = torch.permute(self.embedding(syndromes), (0, 3, 1, 2))
        X = self.conv(X)
        X = X.mean(dim=(2,3))
        return self.proj(X)



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
        return {'decay': self.decay, 'value': self.value}
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.value = state_dict['value']


class RepetitionCodeDataset(IterableDataset):
    """Dataset for repetition code syndrome data with fixed n=25, p=0.07, varying rounds 1-25"""
    
    def __init__(self, batch_size=512):
        super().__init__()
        self.n = 7
        self.p = 0.02
        self.batch_size = batch_size
        self.rounds_range = (1, 25)
    
    def __iter__(self):
        """Generate infinite stream of repetition code data batches."""
        while True:
            rounds = np.random.randint(self.rounds_range[0], self.rounds_range[1] + 1)
            
            # Get circuit and sample data
            circ = get_circ(self.n, rounds, self.p)
            detector_error_model = circ.detector_error_model()
            sampler = detector_error_model.compile_sampler()
            
            # Sample syndromes and observables
            synd, obs, _ = sampler.sample(self.batch_size)
            
            # Reshape syndromes to (batch, rounds, n-1)
            X = synd.reshape((-1, rounds+1, self.n-1))
            
            yield X.astype(int), obs.astype(np.float32)


class RepetitionDecoderTrainer(L.LightningModule):
    def __init__(self, embedding_dim=128, lr=1e-3, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = Decoder(embedding_dim=embedding_dim)
        self.batch_size = batch_size
        self.lr = lr
        
        # Metrics tracking
        self.loss_ema = EMA(0.995)
        self.accuracy_ema = [EMA(0.995) for _ in range(25)]
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        rounds = x.shape[1]-2
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        
        accuracy = ((y_hat>0) != y).float().mean()
        
        # Update EMA metrics
        self.loss_ema.update(loss.item())
        self.accuracy_ema[rounds].update(accuracy.item())
        
        if batch_idx > 150:
            self.log('loss_ema', self.loss_ema.get(), on_step=True, prog_bar=True)
            for i in range(25):
                self.log(f'inacc{i}', self.accuracy_ema[i].get(), on_step=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        dataset = RepetitionCodeDataset(batch_size=self.batch_size)
        return DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)


def train_repetition_decoder():
    """Main training function for repetition code decoder"""
    # Model parameters
    embedding_dim = 64
    lr = 3e-4
    batch_size = 512
    max_steps = 100000
    
    # Create model
    model = RepetitionDecoderTrainer(
        embedding_dim=embedding_dim,
        lr=lr,
        batch_size=batch_size
    )
    
    # Setup wandb logger
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="scaling",
        group="repitition",
        name="repetition_decoder",
        config={
            "n": 25,
            "p": 0.07,
            "rounds_range": "1-25",
            "embedding_dim": embedding_dim,
            "lr": lr,
            "batch_size": batch_size,
            "max_steps": max_steps
        }
    )
    
    # Setup CSV logger
    csv_logger = L.pytorch.loggers.CSVLogger(
        save_dir="./logs",
        name="repetition_decoder"
    )
    
    # Create trainer
    trainer = L.Trainer(
        max_steps=max_steps,
        logger=[wandb_logger, csv_logger],
        log_every_n_steps=50,
        accelerator="auto",
        devices=1
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(model)
    
    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    train_repetition_decoder()
    # collect_matching()

