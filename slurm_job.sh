#!/bin/bash
#SBATCH --job-name=congress-lm-training
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1 
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bettencourt@u.northwestern.edu
echo "Starting job"
set -euo pipefail

echo "Setting project root"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"


echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Project root: $PROJECT_ROOT"
echo "Start time:   $(date)"

module purge all
module load mamba/24.3.0
# Activate conda environment
PYTHON="conda run -n gen-ai-text python"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo "=== Starting BERT training ==="
$PYTHON models/BERT/bert.py
echo "=== BERT training complete ==="

echo "=== Starting GPT-2 training ==="
$PYTHON models/GPT2/gpt2.py
echo "=== GPT-2 training complete ==="

echo "End time: $(date)"
