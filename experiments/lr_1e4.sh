#!/bin/bash
#SBATCH --job-name=lr_1e4
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/n/netscratch/yelin_lab/Everyone/andigu/scaling/lr_1e4/slurm_%j.out
#SBATCH --error=/n/netscratch/yelin_lab/Everyone/andigu/scaling/lr_1e4/slurm_%j.err

set -e

echo "=== Starting Experiment: lr_1e4 ==="
echo "Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: lr_1e4"
echo "Architecture: resnet50"
echo "Embedding Dim: 64"
echo "Learning Rate: 1e-4"
echo ""

# Create experiment output directory
mkdir -p /n/netscratch/yelin_lab/Everyone/andigu/scaling/lr_1e4

# Setup environment
source ~/.bashrc
cd /n/home07/andigu/scale
source .venv/bin/activate
export PYTHONPATH=/n/home07/andigu/scale/src:$PYTHONPATH

# Run training with Hydra config overrides
python src/train.py \
    experiment=lr_1e4

echo ""
echo "=== Experiment Complete: lr_1e4 ==="
echo "Time: $(date)"
