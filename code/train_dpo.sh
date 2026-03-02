#!/bin/bash

#SBATCH --job-name=dpo_qwen_training    # Job name
#SBATCH --output=dpo_training_%j.out      # Standard output log
#SBATCH --error=dpo_training_%j.err       # Standard error log
#SBATCH --nodes=1                         # Run on a single node
#SBATCH --ntasks=1                        # Run a single task
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --time=02:00:00                   # Time limit hrs:min:sec

# --- Environment Setup ---
# python -m venv myenv
# source myenv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install --upgrade -r train_dpo/requirements.txt --quiet

# --- Run the Training Script ---
echo "Starting DPO training script..."

python train_dpo.py

echo "Script finished."
