#!/bin/bash
#SBATCH --job-name=congressional_bert
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/models/bert/logs/slurm_%j.out
#SBATCH --error=/models/bert/logs/slurm_%j.err

mkdir -p /models/bert/logs

conda activate congressional_nlp

echo "Starting BERT training at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

python /models/bert/bert.py

echo "BERT training finished at $(date)"
