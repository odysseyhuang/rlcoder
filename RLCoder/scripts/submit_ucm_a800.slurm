#!/bin/bash
#SBATCH --job-name=rlcoder_ucm
#SBATCH --partition=gpuA800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log_infer/%j_%x.out
#SBATCH --error=log_infer/%j_%x.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/share/home/202421045349/projects/RLCoder/RLCoder}"
CONDA_ENV="${CONDA_ENV:-/share/home/202421045349/conda_envs/rlcoder}"
RUN_MODE="${RUN_MODE:-ucm_debug}"
UCM_RUN_TAG="${UCM_RUN_TAG:-}"

RETRIEVER_MODEL="${RETRIEVER_MODEL:-./models/RLRetriever}"
GENERATOR_MODEL="${GENERATOR_MODEL:-./models/deepseek-coder-6.7b-base}"
RLCODER_MODEL="${RLCODER_MODEL:-./models/unixcoder-base}"

GENERATOR_BATCH_SIZE_PER_GPU="${GENERATOR_BATCH_SIZE_PER_GPU:-8}"
RETRIEVER_BATCH_SIZE_PER_GPU="${RETRIEVER_BATCH_SIZE_PER_GPU:-32}"
NUM_WORKERS="${NUM_WORKERS:-6}"

UCM_BASE_TOPK="${UCM_BASE_TOPK:-0}"
UCM_TOPK_PER_PATH="${UCM_TOPK_PER_PATH:-10}"
UCM_PATH_TOPK="${UCM_PATH_TOPK:-5}"
UCM_CANDIDATE_POOL_SIZE="${UCM_CANDIDATE_POOL_SIZE:-120}"
UCM_ENABLE_PATH_QUERY="${UCM_ENABLE_PATH_QUERY:-0}"
UCM_QUERY_IDENTIFIER_LIMIT="${UCM_QUERY_IDENTIFIER_LIMIT:-64}"
UCM_QUERY_IMPORT_LIMIT="${UCM_QUERY_IMPORT_LIMIT:-32}"

cd "${PROJECT_DIR}"
mkdir -p log_infer result_infer

if [[ -n "${UCM_RUN_TAG}" && "${UCM_RUN_TAG}" =~ [^A-Za-z0-9._-] ]]; then
  echo "Invalid UCM_RUN_TAG=${UCM_RUN_TAG}; use only letters, numbers, dot, underscore, or hyphen." >&2
  exit 2
fi

make_output_dir() {
  local base_dir="$1"
  if [[ -n "${UCM_RUN_TAG}" ]]; then
    printf "%s_%s" "${base_dir}" "${UCM_RUN_TAG}"
  else
    printf "%s" "${base_dir}"
  fi
}

export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Run mode: ${RUN_MODE}"
echo "Run tag: ${UCM_RUN_TAG:-default}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Project dir: ${PROJECT_DIR}"
echo "Start time: $(date)"

COMMON_ARGS=(
  --eval
  --weighted_keywords
  --enable_generation
  --inference_type unixcoder_with_rl
  --retriever_model_path "${RETRIEVER_MODEL}"
  --generator_model_path "${GENERATOR_MODEL}"
  --generator_max_crossfile_length 1536
  --generator_max_context_length 2048
  --generator_batch_size_per_gpu "${GENERATOR_BATCH_SIZE_PER_GPU}"
  --retriever_batch_size_per_gpu "${RETRIEVER_BATCH_SIZE_PER_GPU}"
  --num_workers "${NUM_WORKERS}"
)

UCM_ARGS=(
  --enable_ucm
  --enable_multi_path_retrieval
  --ucm_base_topk "${UCM_BASE_TOPK}"
  --ucm_topk_per_path "${UCM_TOPK_PER_PATH}"
  --ucm_path_topk "${UCM_PATH_TOPK}"
  --ucm_candidate_pool_size "${UCM_CANDIDATE_POOL_SIZE}"
  --ucm_query_identifier_limit "${UCM_QUERY_IDENTIFIER_LIMIT}"
  --ucm_query_import_limit "${UCM_QUERY_IMPORT_LIMIT}"
)

if [[ "${UCM_ENABLE_PATH_QUERY}" == "1" ]]; then
  UCM_ARGS+=(--ucm_enable_path_query)
fi

case "${RUN_MODE}" in
  baseline_debug)
    OUTPUT_DIR="$(make_output_dir "result_infer/RLCoder_baseline_debug")"
    echo "Output dir: ${OUTPUT_DIR}"
    conda run -p "${CONDA_ENV}" python main.py \
      "${COMMON_ARGS[@]}" \
      --debug \
      --output_dir "${OUTPUT_DIR}"
    ;;

  ucm_debug)
    OUTPUT_DIR="$(make_output_dir "result_infer/UCM_multipath_debug")"
    echo "Output dir: ${OUTPUT_DIR}"
    conda run -p "${CONDA_ENV}" python main.py \
      "${COMMON_ARGS[@]}" \
      "${UCM_ARGS[@]}" \
      --debug \
      --ucm_trace_retrieval \
      --output_dir "${OUTPUT_DIR}"
    ;;

  ucm_repocoder_debug)
    OUTPUT_DIR="$(make_output_dir "result_infer/UCM_multipath_repocoder_debug")"
    echo "Output dir: ${OUTPUT_DIR}"
    conda run -p "${CONDA_ENV}" python main.py \
      "${COMMON_ARGS[@]}" \
      "${UCM_ARGS[@]}" \
      --debug \
      --enable_repocoder \
      --rlcoder_model_path "${RLCODER_MODEL}" \
      --ucm_trace_retrieval \
      --output_dir "${OUTPUT_DIR}"
    ;;

  ucm_full)
    OUTPUT_DIR="$(make_output_dir "result_infer/UCM_multipath_deepseekcoder_7b_crossfile_1536_infile_512")"
    echo "Output dir: ${OUTPUT_DIR}"
    conda run -p "${CONDA_ENV}" python main.py \
      "${COMMON_ARGS[@]}" \
      "${UCM_ARGS[@]}" \
      --output_dir "${OUTPUT_DIR}"
    ;;

  *)
    echo "Unknown RUN_MODE=${RUN_MODE}" >&2
    echo "Use one of: baseline_debug, ucm_debug, ucm_repocoder_debug, ucm_full" >&2
    exit 2
    ;;
esac

echo "End time: $(date)"
