#!/bin/bash
#SBATCH --job-name=congress-lm-training
#SBATCH --partition=gpu           # change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1         # change GPU type to match your cluster
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bryant.bettencourt.dev@gmail.com

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p outputs/logs

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Project root: $PROJECT_ROOT"
echo "Start time:   $(date)"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gen-ai-text

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo "=== Starting BERT training ==="
python models/BERT/bert.py
echo "=== BERT training complete ==="

echo "=== Starting GPT-2 training ==="
python models/GPT2/gpt2.py
echo "=== GPT-2 training complete ==="

echo "End time: $(date)"
