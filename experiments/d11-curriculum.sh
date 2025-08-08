#!/bin/bash
#SBATCH --job-name=d11-curriculum
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100"
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/n/netscratch/yelin_lab/Everyone/andigu/scaling/d11-curriculum/slurm_%j.out
#SBATCH --error=/n/netscratch/yelin_lab/Everyone/andigu/scaling/d11-curriculum/slurm_%j.err

set -e

echo "=== Starting Large-Scale Curriculum Learning: d19 ==="
echo "Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: d11-curriculum"
echo "Purpose: Full-scale 3-stage curriculum learning pipeline"
echo "Distance: 19 (production scale)"
echo "GPUs: 4x H100"
echo "Workers: 16 per GPU (64 total)"
echo "Duration: ~24 hours for 500k steps"
echo ""

# Create experiment output directory
mkdir -p /n/netscratch/yelin_lab/Everyone/andigu/scaling/d11-curriculum

# Setup environment
source ~/.bashrc
cd /n/home07/andigu/scale
source .venv/bin/activate
export PYTHONPATH=/n/home07/andigu/scale/src:$PYTHONPATH

echo "=== Environment Setup Complete ==="
echo "Python path: $PYTHONPATH"
echo "Working directory: $(pwd)"
echo "Available GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo ""

# Run 3-stage curriculum learning with production parameters
echo "=== Running Large-Scale Curriculum Learning ==="
echo "Stage 1: p=0.5 for 200,000 steps (low noise, easy learning)"
echo "Stage 2: p=0.5→2.1 over 200,000 steps (curriculum ramp)"  
echo "Stage 3: p=2.1 for 100,000 steps (target noise, final convergence)"
echo "Total: 500,000 steps (~20-24 hours)"
echo ""
echo "Multi-GPU Configuration:"
echo "- 4x H100 GPUs with DDP"
echo "- 16 workers per GPU (64 total DataLoader workers)"
echo "- Synchronized batch normalization"
echo "- Auto-tuned batch size per GPU"
echo ""

python src/train.py \
    experiment=baseline \
    dataset.d=11 \
    dataset.rounds_max=11 \
    dataset.mwpm_filter=false \
    dataset.chunking=[1,1,1] \
    model.architecture=resnet50 \
    model.embedding_dim=128 \
    model.channel_multipliers=[2,2.5,3,3.5] \
    training.lr=4e-4 \
    training.batch_size=null \
    training.log_every_n_steps=100 \
    training.checkpoint_every_minutes=15 \
    training.precision=bf16-mixed \
    training.gradient_clip_val=1.0 \
    training.gradient_clip_algorithm=norm \
    hardware.accelerator=auto \
    hardware.devices=4 \
    hardware.strategy=auto \
    hardware.num_nodes=1 \
    hardware.sync_batchnorm=true \
    hardware.num_workers=16 \
    hardware.prefetch_factor=4 \
    hardware.persistent_workers=true \
    curriculum.enabled=true \
    curriculum.stage1_p=0.5 \
    curriculum.stage1_steps=200000 \
    curriculum.stage2_p_end=2.1 \
    curriculum.stage2_steps=200000 \
    curriculum.stage3_steps=100000 \
    experiment.name=d11-curriculum \
    wandb.enabled=true \
    wandb.project=scaling

echo ""
echo "=== Large-Scale Curriculum Learning Complete ==="
echo "Time: $(date)"
echo ""
echo "=== Validation Checklist ==="
echo "1. Check logs for stage transitions at steps ~200k and ~400k"
echo "2. Verify W&B shows smooth curriculum_p progression: 0.5→2.1"
echo "3. Monitor multi-GPU synchronization and batch size consistency"
echo "4. Confirm checkpoints capture curriculum state correctly"
echo "5. Validate loss/accuracy trends across all three stages"
echo "6. Check for any DDP hanging or worker timeout issues"
echo ""
echo "Expected Timeline:"
echo "- Stage 1 (0-200k steps): ~10 hours"
echo "- Stage 2 (200k-400k steps): ~10 hours" 
echo "- Stage 3 (400k-500k steps): ~5 hours"
echo "Total: ~25 hours including checkpointing overhead"
echo ""