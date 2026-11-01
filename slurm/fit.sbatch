#!/bin/bash
#SBATCH --job-name=mrtt-bayes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --array=0-2

# One array task per condition. Edit the module/venv lines for your cluster.
CONDS=(RND MAX FAIR)
COND=${CONDS[$SLURM_ARRAY_TASK_ID]}

source venv/bin/activate
python fit.py "$COND"
