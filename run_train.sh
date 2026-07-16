#!/bin/bash

# --- SLURM RESOURCE ALLOCATION DIRECTIVES ---
#SBATCH --job-name=osudl-pspnet
#SBATCH --account=dl-course-q2           
#SBATCH --partition=dl-course-q2         
#SBATCH --qos=gpu-xlarge                 
#SBATCH --mem=48G                        
#SBATCH --cpus-per-task=8                
#SBATCH --time=12:00:00                  

# This line tells SLURM to launch 5 parallel, independent jobs (Tasks 0 through 4).
# To run ONLY specific tests, change the array (e.g., --array=0 for just 0%, or --array=3-4 for just 50% and 100%).
#SBATCH --array=0-4  

#SBATCH --gres=gpu:1 --gres=shard:22528

# --- NOTIFICATIONS AND LOGGING ---
# %A is the master job ID, %a is the specific array task ID (0, 1, 2, 3, or 4)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=MRTDGI05D05C351L@studium.unict.it
#SBATCH --output=logs/job-%A_task-%a.log

mkdir -p logs

echo "=== STARTING DEEP LEARNING PROJECT (ARRAY MODE) ==="
echo "Allocated Node: $SLURM_NODELIST"

# Define the array of real data percentages to test
# Index mapping: 0=(0.0), 1=(0.10), 2=(0.25), 3=(0.50), 4=(1.0)
PERCENTAGES=(0.0 0.10 0.25 0.50 1.0)

# Fetch the specific percentage for the current running task
CURRENT_PERC=${PERCENTAGES[$SLURM_ARRAY_TASK_ID]}

echo "Running Task ID: $SLURM_ARRAY_TASK_ID"
echo "Executing test with Real Percentage: $CURRENT_PERC"

# Execution Logic: 
# If the percentage is exactly 0.0, we execute the Baseline (Unsupervised).
# Otherwise, we execute Domain Adaptation with the --supervised flag.
if [ "$CURRENT_PERC" = "0.0" ]; then
    echo "-> Mode: Baseline (0% Real Data, Unsupervised)"
    apptainer run --nv /shared/sifs/latest.sif python models/train.py --real_perc 0.0 --epochs 30
else
    echo "-> Mode: Domain Adaptation ($CURRENT_PERC Real Data, Supervised)"
    apptainer run --nv /shared/sifs/latest.sif python models/train.py --real_perc $CURRENT_PERC --supervised --epochs 30
fi

echo "=== FINISHED TASK $SLURM_ARRAY_TASK_ID ==="