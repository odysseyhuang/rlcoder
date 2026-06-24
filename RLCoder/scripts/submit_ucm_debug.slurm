#!/usr/bin/env bash
#SBATCH --job-name=rlcoder-ucm-debug
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
# Change this when submitting from a different location.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
mkdir -p "${PROJECT_DIR}/slurm_logs"
cd "${PROJECT_DIR}"
# Adjust these two lines to your platform. On the USTC-style platform we used:
module load anaconda/3-2024.02.01 || true
source /share/software/anaconda3/2024.02.01/etc/profile.d/conda.sh
conda activate repocoder
export TOKENIZERS_PARALLELISM=false
export RUN_MODE="${RUN_MODE:-ucm_debug}"
export PROJECT_DIR
bash scripts/run_ucm_job.sh