#!/bin/bash
#SBATCH --job-name=d19-curriculum2
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=/n/netscratch/yelin_lab/Everyone/andigu/scaling/d19-curriculum2/slurm_%j.out
#SBATCH --error=/n/netscratch/yelin_lab/Everyone/andigu/scaling/d19-curriculum2/slurm_%j.err

set -e

echo "=== Starting Large-Scale Curriculum Learning: d19 ==="
echo "Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: d19-curriculum2"
echo "Purpose: Full-scale 3-stage curriculum learning pipeline"
echo "Distance: 19 (production scale)"
echo "GPUs: 8x H100 (2 nodes × 4 GPUs)"
echo "Workers: 16 per GPU (128 total)"
echo "Duration: ~24 hours for 500k steps"
echo ""

# Create experiment output directory
mkdir -p /n/netscratch/yelin_lab/Everyone/andigu/scaling/d19-curriculum2

# Setup environment
source ~/.bashrc
cd /n/home07/andigu/scale
source .venv/bin/activate
export PYTHONPATH=/n/home07/andigu/scale/src:$PYTHONPATH
export NUMBA_CACHE_DIR=/tmp

# Setup multi-node distributed environment variables for PyTorch Lightning
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
# Lightning expects these specific environment variables
export NODE_RANK=$SLURM_NODEID
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

echo "=== Environment Setup Complete ==="
echo "Python path: $PYTHONPATH"
echo "Working directory: $(pwd)"
echo "Available GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Node Rank: $NODE_RANK"
echo ""

# Run 3-stage curriculum learning with production parameters
echo "=== Running Large-Scale Curriculum Learning ==="
echo "Stage 1: p=0.5 for 200,000 steps (low noise, easy learning)"
echo "Stage 2: p=0.5→2.1 over 200,000 steps (curriculum ramp)"  
echo "Stage 3: p=2.1 for 100,000 steps (target noise, final convergence)"
echo "Total: 500,000 steps (~20-24 hours)"
echo ""
echo "Multi-Node Configuration:"
echo "- 2 nodes × 4 H100 GPUs = 8x H100 total"
echo "- 16 workers per GPU (128 total DataLoader workers)"
echo "- Synchronized batch normalization across nodes"
echo "- Auto-tuned batch size per GPU"
echo ""

srun python src/train.py \
    experiment=baseline \
    dataset.d=19 \
    dataset.rounds_max=19 \
    dataset.mwpm_filter=false \
    dataset.chunking=[1,1,1] \
    model.architecture=resnet50 \
    model.embedding_dim=128 \
    model.channel_multipliers=[2,2.5,3,3.5] \
    training.lr=3e-4 \
    training.batch_size=null \
    training.log_every_n_steps=100 \
    training.checkpoint_every_minutes=15 \
    training.precision=bf16-mixed \
    training.gradient_clip_val=1.0 \
    training.gradient_clip_algorithm=norm \
    hardware.accelerator=auto \
    hardware.devices=4 \
    hardware.strategy=ddp \
    hardware.num_nodes=2 \
    hardware.sync_batchnorm=true \
    hardware.num_workers=8 \
    hardware.prefetch_factor=4 \
    hardware.persistent_workers=true \
    curriculum.enabled=true \
    curriculum.stage1_p=0.75 \
    curriculum.stage1_steps=80000 \
    curriculum.stage2_p_end=2.1 \
    curriculum.stage2_steps=200000 \
    curriculum.stage3_steps=500000 \
    experiment.name=d19-curriculum2 \
    wandb.enabled=true \
    wandb.project=scaling \
    hardware.strategy=ddp

echo ""
echo "=== Large-Scale Curriculum Learning Complete ==="
echo "Time: $(date)"
echo ""
echo "=== Validation Checklist ==="
echo "1. Check logs for stage transitions at steps ~200k and ~400k"
echo "2. Verify W&B shows smooth curriculum_p progression: 0.5→2.1"
echo "3. Monitor multi-node/multi-GPU synchronization and batch size consistency"
echo "4. Confirm checkpoints capture curriculum state correctly"
echo "5. Validate loss/accuracy trends across all three stages"
echo "6. Check for any multi-node DDP hanging or worker timeout issues"
echo ""
echo "Expected Timeline:"
echo "- Stage 1 (0-200k steps): ~10 hours"
echo "- Stage 2 (200k-400k steps): ~10 hours" 
echo "- Stage 3 (400k-500k steps): ~5 hours"
echo "Total: ~25 hours including checkpointing overhead"
echo ""
