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
OUTPUT_DIR="result_infer/RLCoder_starcoderbase_7b_crossfile_1536_infile_512"
cd "${PROJECT_DIR}"
mkdir -p log_infer "${OUTPUT_DIR}"
export TOKENIZERS_PARALLELISM=false
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
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
  --generator_max_crossfile_length 1536 \
  --generator_max_context_length 2048 \
  --generator_batch_size_per_gpu 8 \
  --retriever_batch_size_per_gpu 32 \
  --num_workers 6 \
  --output_dir "${OUTPUT_DIR}"
echo "End time: $(date)"
