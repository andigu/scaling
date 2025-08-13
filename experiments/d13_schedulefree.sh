#!/bin/bash
#SBATCH --job-name=d13-schedulefree
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100"
#SBATCH --cpus-per-gpu=9
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/n/netscratch/yelin_lab/Everyone/andigu/scaling/d13-schedulefree/slurm_%j.out
#SBATCH --error=/n/netscratch/yelin_lab/Everyone/andigu/scaling/d13-schedulefree/slurm_%j.err

set -e

echo "=== Starting Training: d13 with ScheduleFree Optimizer ==="
echo "Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: d13-schedulefree"
echo "Purpose: Training d13 surface code with ScheduleFree optimizer"
echo "Distance: 13"
echo "GPU: 1x H100"
echo "Workers: 8 per GPU"
echo ""

# Create experiment output directory
mkdir -p /n/netscratch/yelin_lab/Everyone/andigu/scaling/d13-schedulefree

# Setup environment
source ~/.bashrc
cd /n/home07/andigu/scale
source .venv/bin/activate
export PYTHONPATH=/n/home07/andigu/scale/src:$PYTHONPATH
export NUMBA_CACHE_DIR=/tmp

echo "=== Environment Setup Complete ==="
echo "Python path: $PYTHONPATH"
echo "Working directory: $(pwd)"
echo "Available GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo ""

# Run training with ScheduleFree optimizer
echo "=== Running Training with ScheduleFree Optimizer ==="
echo "Configuration:"
echo "- Model: ResNet50"
echo "- Embedding dim: 128"
echo "- Learning rate: 3e-4"
echo "- Optimizer: ScheduleFree (RAdamScheduleFree)"
echo "- No curriculum learning"
echo "- Max steps: 50000"
echo ""

python src/train.py \
    experiment=baseline \
    dataset.code_type=surface_code \
    dataset.d=13 \
    dataset.rounds_max=13 \
    dataset.p=2.1 \
    dataset.mwpm_filter=false \
    dataset.chunking=[1,1,1] \
    model.architecture=resnet50 \
    model.embedding_dim=64 \
    model.channel_multipliers=[2,4,8,16] \
    model.stage3_stride=[1,1,1] \
    model.stage4_stride=[1,1,1] \
    model.use_lstm=false \
    training.lr=3e-4 \
    training.weight_decay=0.0 \
    training.optimizer=schedulefree \
    training.batch_size=null \
    training.max_steps=50000 \
    training.log_every_n_steps=100 \
    training.checkpoint_every_minutes=15 \
    training.precision=bf16-mixed \
    training.gradient_clip_val=1.0 \
    training.gradient_clip_algorithm=norm \
    training.accumulate_grad_batches=1 \
    hardware.accelerator=auto \
    hardware.devices=1 \
    hardware.strategy=auto \
    hardware.num_nodes=1 \
    hardware.sync_batchnorm=true \
    hardware.num_workers=8 \
    hardware.prefetch_factor=4 \
    hardware.persistent_workers=true \
    curriculum.enabled=false \
    experiment.name=d13-schedulefree \
    experiment.base_dir=/n/netscratch/yelin_lab/Everyone/andigu/scaling \
    wandb.enabled=true \
    wandb.project=scaling \
    wandb.tags=[resnet,surface_code,decoder,d13,schedulefree]

echo ""
echo "=== Training Complete ==="
echo "Time: $(date)"
echo ""
echo "=== Post-Training Checklist ==="
echo "1. Check W&B for training curves and metrics"
echo "2. Verify checkpoints saved correctly"
echo "3. Review loss convergence patterns"
echo "4. Compare with AdamW results"
echo ""