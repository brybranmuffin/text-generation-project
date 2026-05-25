#!/bin/bash
#SBATCH --job-name=congress-lm-training
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1 
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bettencourt@u.northwestern.edu
echo "Starting job"
set -euo pipefail

echo "Setting project root"
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT"

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Project root: $PROJECT_ROOT"
echo "Start time:   $(date)"

module purge all
module load cuda/12.6.2-gcc-12.4.0
module load mamba/24.3.0

export HF_HOME="/scratch/rsr7518/.cache/huggingface"
export HF_DATASETS_CACHE="/scratch/rsr7518/.cache/huggingface/datasets"

PYTHON="conda run -n gen-ai-text python"

echo "Python:         $($PYTHON -c 'import sys; print(sys.executable)')"
echo "PyTorch:        $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"

echo "=== Starting BERT training ==="
$PYTHON models/BERT/bert.py
echo "=== BERT training complete ==="

echo "=== Starting GPT-2 training ==="
$PYTHON models/GPT2/gpt2.py
echo "=== GPT-2 training complete ==="

echo "End time: $(date)"
