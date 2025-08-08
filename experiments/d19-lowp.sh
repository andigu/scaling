#!/bin/bash
#SBATCH --job-name=d19-lowp
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100"
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/n/netscratch/yelin_lab/Everyone/andigu/scaling/d19-lowp/slurm_%j.out
#SBATCH --error=/n/netscratch/yelin_lab/Everyone/andigu/scaling/d19-lowp/slurm_%j.err

set -e

echo "=== Starting Experiment: d19-lowp ==="
echo "Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: d19-lowp"
echo ""

# Create experiment output directory
mkdir -p /n/netscratch/yelin_lab/Everyone/andigu/scaling/d19-lowp

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
    experiment.name=d19-lowp \
    dataset.d=19 \
    dataset.p=0.5 \
    dataset.chunking=[1,1,1] \
    model.channel_multipliers=[2,2.5,3,3.5] \
    model.embedding_dim=128

echo ""
echo "=== Experiment Complete: d19-lowp ==="
echo "Time: $(date)"
