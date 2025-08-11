#!/bin/bash
#SBATCH --job-name=beliefmatch
#SBATCH --partition=serial_requeue
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=32G

source venv/bin/activate
python src/bm.py
