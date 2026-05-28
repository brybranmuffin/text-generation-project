#!/bin/bash
#SBATCH --job-name=bert_activations
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=activations/logs/bert_collect_%j.out
#SBATCH --error=activations/logs/bert_collect_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bettencourt@u.northwestern.edu

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT"

mkdir -p activations/logs

echo "======================================"
echo "BERT Activation Collection"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "======================================"

module purge all
module load cuda/12.6.2-gcc-12.4.0
module load mamba/24.3.0

export HF_HOME="/scratch/rsr7518/.cache/huggingface"
export HF_DATASETS_CACHE="/scratch/rsr7518/.cache/huggingface/datasets"

PYTHON="conda run -n gen-ai-text python"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "======================================"
$PYTHON collect_activations_bert.py \
    --n_speeches 50000 \
    --layers last \
    --batch_size 64 \
    --output_dir ./activations \
    --data_file ./data/raw_data/filtered_speeches.csv \
    --seed 42 \
    --save_metadata True

echo "======================================"
echo "Finished: $(date)"
echo "======================================"
