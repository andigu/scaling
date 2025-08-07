#!/bin/bash
#SBATCH --job-name=resnet50_lstm_1gpu
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100|h200"
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/n/netscratch/yelin_lab/Everyone/andigu/scaling/resnet50_lstm_1gpu/slurm_%j.out
#SBATCH --error=/n/netscratch/yelin_lab/Everyone/andigu/scaling/resnet50_lstm_1gpu/slurm_%j.err

set -e

echo "=== Starting Experiment: resnet50_lstm_1gpu ==="
echo "Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: resnet50_lstm_1gpu"
echo "Architecture: resnet50"
echo "MWPM Filtering: enabled"
echo "Workers: 15"
echo ""

# Create experiment output directory
mkdir -p /n/netscratch/yelin_lab/Everyone/andigu/scaling/resnet50_lstm_1gpu

# Setup environment
source ~/.bashrc
cd /n/home07/andigu/scale
source .venv/bin/activate
export PYTHONPATH=/n/home07/andigu/scale/src:$PYTHONPATH

# Run training with Hydra config overrides
python src/train.py \
    experiment=baseline \
    model.architecture=resnet50 \
    dataset.mwpm_filter=true \
    hardware.num_workers=15 \
    experiment.name=resnet50_lstm_1gpu \
    model.use_lstm=true \
    dataset.d=19

echo ""
echo "=== Experiment Complete: resnet50_lstm_1gpu ==="
echo "Time: $(date)"
