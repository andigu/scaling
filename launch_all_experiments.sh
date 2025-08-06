#!/bin/bash
#
# Launch all experiment SBATCH scripts
#

set -e

echo "=== Launching All Experiments ==="
echo "Time: $(date)"
echo "User: $USER"
echo ""

# Generate the experiment scripts first
echo "Generating experiment scripts..."
python generate_experiment_scripts.py

echo ""
echo "Submitting all experiments..."

# Count total experiments
total_experiments=$(find experiments/ -name "*.sh" | wc -l)
submitted=0

# Submit each experiment
for script in experiments/*.sh; do
    if [ -f "$script" ]; then
        experiment_name=$(basename "$script" .sh)
        echo "Submitting: $experiment_name"
        
        # Submit the job and capture job ID
        job_id=$(sbatch "$script" | awk '{print $4}')
        echo "  Job ID: $job_id"
        
        submitted=$((submitted + 1))
    fi
done

echo ""
echo "=== Submission Complete ==="
echo "Submitted: $submitted/$total_experiments experiments"
echo "Check job status with: squeue -u $USER"
echo "Monitor experiment outputs in: /n/netscratch/yelin_lab/Everyone/andigu/scaling/"
echo "Monitor in W&B at: https://wandb.ai"
echo ""
echo "Individual experiment directories:"
echo "  /n/netscratch/yelin_lab/Everyone/andigu/scaling/{experiment_name}/"
echo "    ├── checkpoints/     # Lightning checkpoints"
echo "    ├── logs/           # Training logs"
echo "    ├── wandb/          # W&B files"
echo "    ├── metadata.pkl    # Resume metadata"
echo "    └── slurm_*.out     # SLURM output logs"