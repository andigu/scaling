#!/bin/bash
#SBATCH --job-name=no-mult
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100"
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/n/netscratch/yelin_lab/Everyone/andigu/scaling/no-mult/slurm_%j.out
#SBATCH --error=/n/netscratch/yelin_lab/Everyone/andigu/scaling/no-mult/slurm_%j.err

set -e

echo "=== Starting Experiment: no-mult ==="
echo "Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: no-mult"
echo "Architecture: resnet101"
echo "MWPM Filtering: disabled"
echo "Workers: 16"
echo ""

# Create experiment output directory
mkdir -p /n/netscratch/yelin_lab/Everyone/andigu/scaling/no-mult

# Setup environment
source ~/.bashrc
cd /n/home07/andigu/scale
source .venv/bin/activate
export PYTHONPATH=/n/home07/andigu/scale/src:$PYTHONPATH

# Run training with Hydra config overrides
python src/train.py \
    experiment=baseline \
    dataset.mwpm_filter=false \
    hardware.num_workers=16 \
    experiment.name=no-mult \
    dataset.d=9 \
    dataset.p=1.75 \
    dataset.chunking=[1,1,1] \
    model.channel_multipliers=[1,1,1,1]

echo ""
echo "=== Experiment Complete: no-mult ==="
echo "Time: $(date)"
