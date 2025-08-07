#!/bin/bash
#SBATCH --job-name=101_down3
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-gpu=9
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/n/netscratch/yelin_lab/Everyone/andigu/scaling/arch_101_down3/slurm_%j.out
#SBATCH --error=/n/netscratch/yelin_lab/Everyone/andigu/scaling/arch_101_down3/slurm_%j.err

set -e

echo "=== Starting Experiment: arch_101_downsample ==="
echo "Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: arch_101"
echo "Architecture: resnet101"
echo "Embedding Dim: 64"
echo "Learning Rate: 3e-4"
echo ""

# Create experiment output directory
mkdir -p /n/netscratch/yelin_lab/Everyone/andigu/scaling/arch_101_down3

# Setup environment
source ~/.bashrc
cd /n/home07/andigu/scale
source .venv/bin/activate
export PYTHONPATH=/n/home07/andigu/scale/src:$PYTHONPATH

# Run training with Hydra config overrides
python src/train.py \
    experiment=arch_101_down3

echo ""
echo "=== Experiment Complete: arch_101 ==="
echo "Time: $(date)"
