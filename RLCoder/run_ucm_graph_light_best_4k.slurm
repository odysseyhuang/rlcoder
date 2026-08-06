#!/bin/bash
#SBATCH --job-name=ucm_graph_4k
#SBATCH --partition=gpuA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log_infer/%j_ucm_graph_4k.out
#SBATCH --error=log_infer/%j_ucm_graph_4k.err

set -euo pipefail

PROJECT_DIR="/share/home/202421045349/projects/RLCoder/RLCoder"
CONDA_ENV="/share/home/202421045349/conda_envs/rlcoder"
OUTPUT_DIR="result_infer/UCMGraphLightBest_4k"

cd "${PROJECT_DIR}"
mkdir -p log_infer result_infer

export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Start time: $(date)"

conda run -p "${CONDA_ENV}" python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable on this job node'; print('GPU:', torch.cuda.get_device_name(0))"

conda run -p "${CONDA_ENV}" python main.py \
  --eval \
  --weighted_keywords \
  --enable_generation \
  --inference_type unixcoder_with_rl \
  --retriever_model_path ./models/RLRetriever \
  --generator_model_path ./models/deepseek-coder-6.7b-base \
  --generator_max_crossfile_length 3072 \
  --generator_max_context_length 4096 \
  --generator_batch_size_per_gpu 4 \
  --retriever_batch_size_per_gpu 32 \
  --num_workers 6 \
  --enable_ucm \
  --enable_multi_path_retrieval \
  --ucm_base_topk 60 \
  --ucm_topk_per_path 30 \
  --ucm_path_topk 5 \
  --ucm_candidate_pool_size 140 \
  --enable_context_graph \
  --ucm_graph_max_seed 20 \
  --ucm_graph_max_neighbors_per_seed 2 \
  --ucm_graph_max_expanded 40 \
  --ucm_graph_enable_identifier_edges \
  --ucm_graph_enable_import_edges \
  --ucm_graph_enable_api_call_edges \
  --ucm_graph_identifier_max_df 10 \
  --ucm_graph_identifier_query_only \
  --ucm_graph_api_call_max_df 20 \
  --ucm_graph_api_call_weight 1.6 \
  --ucm_graph_import_weight 1.4 \
  --ucm_graph_identifier_weight 1.2 \
  --ucm_graph_same_file_weight 1.0 \
  --ucm_graph_seed_rank_decay 0.05 \
  --ucm_graph_distance_decay 0.75 \
  --ucm_graph_query_overlap_bonus 2 \
  --ucm_graph_query_api_bonus 2 \
  --run_short_name UCMGraphLightBest_4k \
  --output_dir "${OUTPUT_DIR}"

echo "End time: $(date)"