#!/bin/bash
#SBATCH --job-name=eval_pspnet
#SBATCH --account=dl-course-q2           
#SBATCH --partition=dl-course-q2         
#SBATCH --qos=gpu-xlarge                 
#SBATCH --mem=16G                        
#SBATCH --cpus-per-task=4                
#SBATCH --time=04:00:00                  

#SBATCH --gres=gpu:1 --gres=shard:22528

# Output
#SBATCH --output=logs/eval-%j.log
#SBATCH --error=logs/eval-%j.err

echo "=== AVVIO VALUTAZIONE MODELLI ==="
echo "Nodo allocato: $SLURM_NODELIST"

apptainer run --nv /shared/sifs/latest.sif python utility/evaluate.py

echo "=== FINE VALUTAZIONE ==="