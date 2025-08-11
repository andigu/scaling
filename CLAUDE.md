# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantum error correction research project focused on scaling machine learning decoders for surface codes. It uses 3D ResNet architectures with PyTorch Lightning to train neural decoders that predict logical errors from syndrome measurements. The project supports large-scale distributed training with multi-node SLURM orchestration and curriculum learning.

## Key Dependencies and Architecture

- **Python 3.12+** required
- **Package manager**: Uses `uv` (see uv.lock file)
- **ML stack**: PyTorch, PyTorch Lightning, PyTorch Geometric
- **QEC libraries**: Stim (circuit simulation), PyMatching (classical decoding), BeliefMatching
- **Experiment management**: Hydra (config), Wandb (logging), SLURM (compute)
- **Hardware**: Optimized for multi-GPU H100 systems with large memory

## Core Architecture

The codebase implements a 3D ResNet decoder for surface code error correction:

1. **Surface Code Simulation** (`src/simulate/`):
   - Comprehensive quantum error correction code library
   - Surface codes, color codes, CSS codes with noise models
   - Uses Stim for efficient circuit simulation and measurement tracking

2. **3D ResNet Decoder** (`src/resnet.py`):
   - Configurable architectures: ResNet18/34/50/101/152
   - 3D convolutions process (time, height, width) syndrome data
   - Channel multipliers allow memory-efficient variants
   - Pre-activation residual blocks with GELU activation

3. **Training Pipeline** (`src/train.py`):
   - PyTorch Lightning module with distributed training support
   - Automatic batch size tuning and gradient accumulation
   - Curriculum learning with 3-stage noise progression
   - SLURM requeuing support for long experiments

4. **Data Management** (`src/data_module.py`, `src/dataset.py`):
   - Streaming surface code syndrome generation
   - Chunked processing for memory efficiency
   - MWPM filtering for hard negative mining

## Development Commands

```bash
# Install dependencies
uv sync

# Single-GPU training (auto-detects config)
python src/train.py

# Multi-GPU training with specific config
python src/train.py experiment=baseline dataset.d=11 hardware.devices=4

# Run experiment scripts (SLURM)
sbatch experiments/d11-curriculum.sh

# Generate experiment variations
python generate_experiment_scripts.py

# Benchmark MWPM vs neural decoder
python src/mwpm.py
```

## Configuration System

Uses Hydra with structured configs in `configs/`:
- **Base config**: `config.yaml` (hardware, training, model params)
- **Experiments**: `experiment/*.yaml` (architecture variants)
- **Override examples**: `dataset.d=15 model.embedding_dim=256 training.lr=3e-4`

## Experiment Management

- **State management**: Each experiment gets isolated directory with checkpoints, logs, metadata
- **Resumability**: Supports SLURM requeuing with automatic state recovery
- **Multi-node**: Handles W&B run ID sharing and batch size coordination across nodes
- **Curriculum learning**: 3-stage noise progression (low→target noise) with state preservation

## Key File Structure

- `src/train.py`: Main training entry point with Lightning module
- `src/resnet.py`: 3D ResNet decoder architecture
- `src/simulate/`: Comprehensive quantum error correction simulation library
- `src/data_module.py`: Lightning DataModule for streaming syndrome data
- `experiments/*.sh`: SLURM job scripts for large-scale experiments
- `configs/`: Hydra configuration system

## Training Patterns

- **Batch sizing**: Auto-tuning on rank 0, file-sharing to other ranks
- **Checkpointing**: Time-based (every 15min) with full state including EMA metrics
- **Logging**: CSV + W&B with multi-GPU synchronization
- **Curriculum**: Optional 3-stage learning with automatic stage transitions