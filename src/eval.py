#!/usr/bin/env python3
"""
Quick and dirty evaluation script for quantum error correction models.
"""

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from resnet import ResNet3D
from dataset import TemporalSurfaceCodeDataset


def load_model_from_checkpoint(checkpoint_path):
    """Load model from Lightning checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters from checkpoint
    hparams = ckpt.get('hyper_parameters', {})
    
    # Get model config with fallbacks
    cfg = hparams.get('cfg', {})
    model_cfg = cfg.get('model', {})
    
    architecture = model_cfg.get('architecture', 'resnet50')
    embedding_dim = model_cfg.get('embedding_dim', 64)
    stage3_stride = tuple(model_cfg.get('stage3_stride', [1, 1, 1]))
    stage4_stride = tuple(model_cfg.get('stage4_stride', [1, 1, 1]))
    use_lstm = model_cfg.get('use_lstm', False)
    use_chunking = cfg.get('dataset', {}).get('use_chunking', True)
    
    # Create model
    model = ResNet3D(
        architecture=architecture,
        embedding_dim=embedding_dim,
        stage3_stride=stage3_stride,
        stage4_stride=stage4_stride,
        use_lstm=use_lstm,
        use_chunking=use_chunking
    )
    
    # Load weights
    state_dict = ckpt['state_dict']
    model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    model.load_state_dict(model_state)
    model.eval()
    
    return model


def evaluate_model(model, d, p, rounds_max, num_batches=10, batch_size=32):
    """Evaluate model on test data."""
    # Create dataset
    dataset = TemporalSurfaceCodeDataset(
        d=d,
        rounds_max=rounds_max,
        p=p,
        batch_size=batch_size,
        mwpm_filter=False,  # Disable filtering for evaluation
        use_chunking=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # Dataset handles batching
        shuffle=False,
        num_workers=0     # Single-threaded for simplicity
    )
    
    total_correct = 0
    total_samples = 0
    
    print(f"Evaluating on d={d}, p={p}, rounds_max={rounds_max}")
    print(f"Running {num_batches} batches of size {batch_size}")
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    cnt = 0
    with torch.inference_mode():
        for batch_idx, (x, y) in enumerate(dataloader):
            if x.shape[1]-1 != rounds_max: # Ensure correct number of rounds (stupid hack, fix in dataset later to allow for fixed number of rounds)
                continue
            if cnt >= num_batches:
                break
            cnt += 1
                
            # Forward pass
            x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
            pred = model(x).detach().cpu()
            
            # Calculate accuracy
            predictions = (pred > 0).float()
            correct = (predictions == y).sum().item()
            total_correct += correct
            total_samples += y.numel()
                
    overall_acc = total_correct / total_samples
    inacc = 1-overall_acc
    inacc_err = np.sqrt(overall_acc * (1 - overall_acc) / total_samples)
    print(f"\nOverall inaccuracy: {inacc:.4f} ± {inacc_err:.4f}")
    
    return overall_acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate quantum error correction model')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--d', type=int, required=True, help='Surface code distance')
    parser.add_argument('--p', type=float, required=True, help='Error probability parameter') 
    parser.add_argument('--rounds_max', type=int, required=True, help='Maximum number of rounds')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to evaluate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint)
    
    for r in range(1, args.rounds_max + 1):
        accuracy = evaluate_model(
            model, 
            args.d, 
            args.p, 
            r,
            args.num_batches,
            args.batch_size
        )


if __name__ == "__main__":
    main()