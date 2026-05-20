#!/bin/bash
#SBATCH --job-name=congressional_gpt2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/models/gpt2/logs/slurm_%j.out
#SBATCH --error=/models/gpt2/logs/slurm_%j.err

mkdir -p /models/gpt2/logs

conda activate congressional_nlp

echo "Starting GPT-2 training at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

python /models/gpt2/gpt2.py

echo "GPT-2 training finished at $(date)"
