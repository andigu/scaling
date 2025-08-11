#!/bin/bash

#SBATCH --partition=gpu_requeue
#SBATCH --cpus-per-gpu=9
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100"
#SBATCH --time=04:00:00
#SBATCH --job-name=curriculum_resume

# Load environment
source ~/.bashrc

# Navigate to project directory
cd /n/home07/andigu/scale
source .venv/bin/activate
cd src
# Run curriculum training with automatic resume
python distance_curriculum.py --mode curriculum