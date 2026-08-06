#!/bin/bash
#SBATCH --job-name=rlcoder_base_4k
#SBATCH --partition=gpuA800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log_infer/%j_baseline_4k.out
#SBATCH --error=log_infer/%j_baseline_4k.err
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/share/home/202421045349/projects/RLCoder/RLCoder}"
CONDA_ENV="${CONDA_ENV:-/share/home/202421045349/conda_envs/rlcoder}"
GENERATOR_MODEL_PATH="${GENERATOR_MODEL_PATH:-./models/deepseek-coder-6.7b-base}"
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:-./models/RLRetriever}"
GENERATOR_MAX_CONTEXT_LENGTH="${GENERATOR_MAX_CONTEXT_LENGTH:-4096}"
GENERATOR_MAX_CROSSFILE_LENGTH="${GENERATOR_MAX_CROSSFILE_LENGTH:-3072}"
GENERATOR_BATCH_SIZE_PER_GPU="${GENERATOR_BATCH_SIZE_PER_GPU:-8}"
RETRIEVER_BATCH_SIZE_PER_GPU="${RETRIEVER_BATCH_SIZE_PER_GPU:-32}"
NUM_WORKERS="${NUM_WORKERS:-6}"
INFILE_LENGTH=$((GENERATOR_MAX_CONTEXT_LENGTH - GENERATOR_MAX_CROSSFILE_LENGTH))
OUTPUT_DIR="result_infer/RLCoder_deepseekcoder_7b_crossfile_${GENERATOR_MAX_CROSSFILE_LENGTH}_infile_${INFILE_LENGTH}"
if [[ "${GENERATOR_MAX_CONTEXT_LENGTH}" -le 0 || "${GENERATOR_MAX_CROSSFILE_LENGTH}" -le 0 ]]; then
  echo "Generator prompt lengths must be positive integers." >&2
  exit 2
fi
if [[ "${GENERATOR_MAX_CROSSFILE_LENGTH}" -ge "${GENERATOR_MAX_CONTEXT_LENGTH}" ]]; then
  echo "GENERATOR_MAX_CROSSFILE_LENGTH must be smaller than GENERATOR_MAX_CONTEXT_LENGTH." >&2
  exit 2
fi
cd "${PROJECT_DIR}"
mkdir -p log_infer "${OUTPUT_DIR}"
export TOKENIZERS_PARALLELISM=false
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Generator model: ${GENERATOR_MODEL_PATH}"
echo "Retriever model: ${RETRIEVER_MODEL_PATH}"
echo "Generator prompt window: context=${GENERATOR_MAX_CONTEXT_LENGTH}, crossfile=${GENERATOR_MAX_CROSSFILE_LENGTH}, infile=${INFILE_LENGTH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Start time: $(date)"
conda run -p "${CONDA_ENV}" python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable on this job node'; print('GPU:', torch.cuda.get_device_name(0))"
conda run -p "${CONDA_ENV}" python main.py \
  --eval \
  --weighted_keywords \
  --enable_generation \
  --inference_type unixcoder_with_rl \
  --retriever_model_path "${RETRIEVER_MODEL_PATH}" \
  --generator_model_path "${GENERATOR_MODEL_PATH}" \
  --generator_max_crossfile_length "${GENERATOR_MAX_CROSSFILE_LENGTH}" \
  --generator_max_context_length "${GENERATOR_MAX_CONTEXT_LENGTH}" \
  --generator_batch_size_per_gpu "${GENERATOR_BATCH_SIZE_PER_GPU}" \
  --retriever_batch_size_per_gpu "${RETRIEVER_BATCH_SIZE_PER_GPU}" \
  --num_workers "${NUM_WORKERS}" \
  --output_dir "${OUTPUT_DIR}"
echo "End time: $(date)"
