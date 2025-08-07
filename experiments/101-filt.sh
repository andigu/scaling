#!/bin/bash
#SBATCH --job-name=resnet101_filt_4gpu
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/n/netscratch/yelin_lab/Everyone/andigu/scaling/resnet101_filt_4gpu/slurm_%j.out
#SBATCH --error=/n/netscratch/yelin_lab/Everyone/andigu/scaling/resnet101_filt_4gpu/slurm_%j.err

set -e

echo "=== Starting Experiment: resnet101_filt_4gpu ==="
echo "Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: resnet101_filt_4gpu"
echo "Architecture: resnet101"
echo "MWPM Filtering: enabled"
echo "Workers: 15"
echo ""

# Create experiment output directory
mkdir -p /n/netscratch/yelin_lab/Everyone/andigu/scaling/resnet101_filt_4gpu

# Setup environment
source ~/.bashrc
cd /n/home07/andigu/scale
source .venv/bin/activate
export PYTHONPATH=/n/home07/andigu/scale/src:$PYTHONPATH

# Run training with Hydra config overrides
python src/train.py \
    experiment=arch_101 \
    model.architecture=resnet101 \
    dataset.mwpm_filter=true \
    hardware.num_workers=15 \
    experiment.name=resnet101_filt_4gpu \
    dataset.d=19

echo ""
echo "=== Experiment Complete: resnet101_filt_4gpu ==="
echo "Time: $(date)"
