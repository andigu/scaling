#!/usr/bin/env python3
"""
Generate individual SBATCH scripts for each experiment.
This replaces the complex Hydra multirun approach with simple native SLURM requeuing.
"""

import os
import yaml
from pathlib import Path
from omegaconf import OmegaConf

def load_experiment_config(experiment_name: str) -> dict:
    """Load and merge experiment config with base config."""
    
    # Load base config
    base_config_path = Path("configs/config.yaml")
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load experiment config
    exp_config_path = Path(f"configs/experiment/{experiment_name}.yaml")
    with open(exp_config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    # Merge configs (experiment overrides base)
    merged_config = base_config.copy()
    
    # Deep merge the configs
    if 'model' in exp_config:
        merged_config.setdefault('model', {}).update(exp_config['model'])
    if 'training' in exp_config:
        merged_config.setdefault('training', {}).update(exp_config['training'])
    if 'experiment' in exp_config:
        merged_config.setdefault('experiment', {}).update(exp_config['experiment'])
    
    return merged_config

def generate_sbatch_script(experiment_name: str, config: dict) -> str:
    """Generate SBATCH script content for an experiment."""
    
    # Extract key parameters for display
    architecture = config['model']['architecture']
    embedding_dim = config['model']['embedding_dim'] 
    lr = config['training']['lr']
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={experiment_name}
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/n/netscratch/yelin_lab/Everyone/andigu/scaling/{experiment_name}/slurm_%j.out
#SBATCH --error=/n/netscratch/yelin_lab/Everyone/andigu/scaling/{experiment_name}/slurm_%j.err

set -e

echo "=== Starting Experiment: {experiment_name} ==="
echo "Time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: {experiment_name}"
echo "Architecture: {architecture}"
echo "Embedding Dim: {embedding_dim}"
echo "Learning Rate: {lr}"
echo ""

# Create experiment output directory
mkdir -p /n/netscratch/yelin_lab/Everyone/andigu/scaling/{experiment_name}

# Setup environment
source ~/.bashrc
cd /n/home07/andigu/scale
source .venv/bin/activate
export PYTHONPATH=/n/home07/andigu/scale/src:$PYTHONPATH

# Run training with Hydra config overrides
python src/train.py \\
    experiment={experiment_name}

echo ""
echo "=== Experiment Complete: {experiment_name} ==="
echo "Time: $(date)"
"""
    
    return script_content

def main():
    """Generate all experiment SBATCH scripts."""
    
    # Create experiments directory
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    
    # Get list of all experiment configs
    experiment_configs = list(Path("configs/experiment").glob("*.yaml"))
    experiment_names = [f.stem for f in experiment_configs]
    
    print(f"Generating SBATCH scripts for {len(experiment_names)} experiments:")
    
    for experiment_name in experiment_names:
        print(f"  - {experiment_name}")
        
        # Load experiment configuration
        config = load_experiment_config(experiment_name)
        
        # Generate SBATCH script
        script_content = generate_sbatch_script(experiment_name, config)
        
        # Write script file
        script_path = experiments_dir / f"{experiment_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    print(f"\\nGenerated {len(experiment_names)} SBATCH scripts in experiments/")
    print(f"To launch all experiments: bash launch_all_experiments.sh")

if __name__ == "__main__":
    main()