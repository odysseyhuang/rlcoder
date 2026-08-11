#!/bin/bash
#SBATCH --job-name=rlcoder_starcoderbase
#SBATCH --partition=gpuA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log_infer/%j_starcoderbase.out
#SBATCH --error=log_infer/%j_starcoderbase.err

set -euo pipefail

PROJECT_DIR="/share/home/202421045349/projects/RLCoder/RLCoder"
CONDA_ENV="/share/home/202421045349/conda_envs/rlcoder"
GENERATOR_MAX_CONTEXT_LENGTH="${GENERATOR_MAX_CONTEXT_LENGTH:-4096}"
GENERATOR_MAX_CROSSFILE_LENGTH="${GENERATOR_MAX_CROSSFILE_LENGTH:-3072}"
GENERATOR_MAX_GENERATION_LENGTH="${GENERATOR_MAX_GENERATION_LENGTH:-64}"
RETRIEVER_QUERY_CONTEXT_LENGTH="${RETRIEVER_QUERY_CONTEXT_LENGTH:-256}"
RETRIEVER_CANDIDATE_CONTEXT_LENGTH="${RETRIEVER_CANDIDATE_CONTEXT_LENGTH:-512}"
SAMPLE_NUMBER="${SAMPLE_NUMBER:-10}"
INFILE_LENGTH=$((GENERATOR_MAX_CONTEXT_LENGTH - GENERATOR_MAX_CROSSFILE_LENGTH))
OUTPUT_DIR="result_infer/RLCoder_starcoderbase_7b_crossfile_${GENERATOR_MAX_CROSSFILE_LENGTH}_infile_${INFILE_LENGTH}"

cd "${PROJECT_DIR}"
mkdir -p log_infer "${OUTPUT_DIR}"

export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Generator model: ./models/starcoderbase-7b"
echo "Generator prompt window: context=${GENERATOR_MAX_CONTEXT_LENGTH}, crossfile=${GENERATOR_MAX_CROSSFILE_LENGTH}, infile=${INFILE_LENGTH}"
echo "Generation/retrieval budget: generation=${GENERATOR_MAX_GENERATION_LENGTH}, retriever_query=${RETRIEVER_QUERY_CONTEXT_LENGTH}, retriever_candidate=${RETRIEVER_CANDIDATE_CONTEXT_LENGTH}, sample/topk=${SAMPLE_NUMBER}"
echo "Start time: $(date)"

conda run -p "${CONDA_ENV}" python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable on this job node'; print('GPU:', torch.cuda.get_device_name(0))"

conda run -p "${CONDA_ENV}" python main.py \
  --eval \
  --weighted_keywords \
  --enable_generation \
  --inference_type unixcoder_with_rl \
  --retriever_model_path ./models/RLRetriever \
  --generator_model_path ./models/starcoderbase-7b \
  --generator_max_crossfile_length "${GENERATOR_MAX_CROSSFILE_LENGTH}" \
  --generator_max_context_length "${GENERATOR_MAX_CONTEXT_LENGTH}" \
  --generator_max_generation_length "${GENERATOR_MAX_GENERATION_LENGTH}" \
  --generator_batch_size_per_gpu 8 \
  --retriever_query_context_length "${RETRIEVER_QUERY_CONTEXT_LENGTH}" \
  --retriever_candidate_context_length "${RETRIEVER_CANDIDATE_CONTEXT_LENGTH}" \
  --retriever_batch_size_per_gpu 32 \
  --sample_number "${SAMPLE_NUMBER}" \
  --num_workers 6 \
  --output_dir "${OUTPUT_DIR}"

echo "End time: $(date)"
