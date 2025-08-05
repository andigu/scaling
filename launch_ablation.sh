#!/bin/bash
#
# Launch ablation sweep with proper SLURM configuration
#

set -e

echo "=== Launching Ablation Sweep ==="
echo "Time: $(date)"
echo "User: $USER"
echo ""

# Activate environment
source .venv/bin/activate

# Launch with proper Hydra SLURM configuration
python src/train.py \
  -m \
  experiment=baseline,arch_18,arch_34,arch_101,arch_152,embed_32,embed_128,embed_256,embed_512,lr_1e4,lr_1e3,lr_3e3,s3_temp,s3_spat,s3_both,s4_temp,s4_spat,s4_both \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition=gpu_requeue \
  hydra.launcher.cpus_per_task=8 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=32 \
  hydra.launcher.timeout_min=480 \
  'hydra.launcher.setup=["source ~/.bashrc","source /n/home07/andigu/scale/.venv/bin/activate","cd /n/home07/andigu/scale","export PYTHONPATH=/n/home07/andigu/scale/src:$PYTHONPATH"]' \
  +hydra.launcher.additional_parameters.constraint=h100

echo ""
echo "=== Submission Complete ==="
echo "Check job status with: squeue -u $USER"
echo "Monitor in W&B at: https://wandb.ai"