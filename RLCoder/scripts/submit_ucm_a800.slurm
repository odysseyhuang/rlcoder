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
RUN_SHORT_NAME="${RUN_SHORT_NAME:-}"

RETRIEVER_MODEL="${RETRIEVER_MODEL:-./models/RLRetriever}"
GENERATOR_MODEL="${GENERATOR_MODEL:-./models/deepseek-coder-6.7b-base}"
RLCODER_MODEL="${RLCODER_MODEL:-./models/unixcoder-base}"

GENERATOR_BATCH_SIZE_PER_GPU="${GENERATOR_BATCH_SIZE_PER_GPU:-8}"
GENERATOR_MAX_CONTEXT_LENGTH="${GENERATOR_MAX_CONTEXT_LENGTH:-2048}"
GENERATOR_MAX_CROSSFILE_LENGTH="${GENERATOR_MAX_CROSSFILE_LENGTH:-1536}"
RETRIEVER_BATCH_SIZE_PER_GPU="${RETRIEVER_BATCH_SIZE_PER_GPU:-32}"
NUM_WORKERS="${NUM_WORKERS:-6}"

UCM_BASE_TOPK="${UCM_BASE_TOPK:-60}"
UCM_TOPK_PER_PATH="${UCM_TOPK_PER_PATH:-30}"
UCM_PATH_TOPK="${UCM_PATH_TOPK:-5}"
UCM_CANDIDATE_POOL_SIZE="${UCM_CANDIDATE_POOL_SIZE:-140}"
UCM_DISABLE_IDENTIFIER_QUERY="${UCM_DISABLE_IDENTIFIER_QUERY:-0}"
UCM_DISABLE_IMPORT_API_QUERY="${UCM_DISABLE_IMPORT_API_QUERY:-0}"
UCM_ENABLE_PATH_QUERY="${UCM_ENABLE_PATH_QUERY:-0}"
UCM_QUERY_IDENTIFIER_LIMIT="${UCM_QUERY_IDENTIFIER_LIMIT:-64}"
UCM_QUERY_IMPORT_LIMIT="${UCM_QUERY_IMPORT_LIMIT:-32}"
UCM_DISABLE_ENHANCED_BM25="${UCM_DISABLE_ENHANCED_BM25:-0}"
UCM_ENABLE_CONTEXT_GATE="${UCM_ENABLE_CONTEXT_GATE:-0}"
UCM_GATE_MAX_AUXILIARY_BLOCKS="${UCM_GATE_MAX_AUXILIARY_BLOCKS:-2}"
UCM_GATE_ALLOW_PATH_ONLY="${UCM_GATE_ALLOW_PATH_ONLY:-0}"
UCM_GATE_STOP_RANK_THRESHOLD="${UCM_GATE_STOP_RANK_THRESHOLD:-2}"
UCM_ENABLE_CONTEXT_GRAPH="${UCM_ENABLE_CONTEXT_GRAPH:-0}"
UCM_GRAPH_MAX_SEED="${UCM_GRAPH_MAX_SEED:-20}"
UCM_GRAPH_MAX_NEIGHBORS_PER_SEED="${UCM_GRAPH_MAX_NEIGHBORS_PER_SEED:-2}"
UCM_GRAPH_MAX_EXPANDED="${UCM_GRAPH_MAX_EXPANDED:-40}"
UCM_GRAPH_ENABLE_IDENTIFIER_EDGES="${UCM_GRAPH_ENABLE_IDENTIFIER_EDGES:-0}"
UCM_GRAPH_ENABLE_IMPORT_EDGES="${UCM_GRAPH_ENABLE_IMPORT_EDGES:-0}"
UCM_GRAPH_ENABLE_API_CALL_EDGES="${UCM_GRAPH_ENABLE_API_CALL_EDGES:-0}"
UCM_GRAPH_IDENTIFIER_MAX_DF="${UCM_GRAPH_IDENTIFIER_MAX_DF:-20}"
UCM_GRAPH_API_CALL_MAX_DF="${UCM_GRAPH_API_CALL_MAX_DF:-20}"
UCM_GRAPH_SAME_FILE_DIRECTION="${UCM_GRAPH_SAME_FILE_DIRECTION:-both}"
UCM_GRAPH_IDENTIFIER_QUERY_ONLY="${UCM_GRAPH_IDENTIFIER_QUERY_ONLY:-0}"
UCM_GRAPH_API_CALL_QUERY_ONLY="${UCM_GRAPH_API_CALL_QUERY_ONLY:-0}"
UCM_GRAPH_SEED_RANK_DECAY="${UCM_GRAPH_SEED_RANK_DECAY:-0.05}"
UCM_GRAPH_DISTANCE_DECAY="${UCM_GRAPH_DISTANCE_DECAY:-0.75}"
UCM_GRAPH_SAME_FILE_WEIGHT="${UCM_GRAPH_SAME_FILE_WEIGHT:-1.0}"
UCM_GRAPH_IDENTIFIER_WEIGHT="${UCM_GRAPH_IDENTIFIER_WEIGHT:-1.2}"
UCM_GRAPH_IMPORT_WEIGHT="${UCM_GRAPH_IMPORT_WEIGHT:-1.4}"
UCM_GRAPH_API_CALL_WEIGHT="${UCM_GRAPH_API_CALL_WEIGHT:-1.6}"
UCM_GRAPH_QUERY_OVERLAP_BONUS="${UCM_GRAPH_QUERY_OVERLAP_BONUS:-2}"
UCM_GRAPH_QUERY_API_BONUS="${UCM_GRAPH_QUERY_API_BONUS:-2}"
UCM_GRAPH_ENABLE_RERANK="${UCM_GRAPH_ENABLE_RERANK:-0}"
UCM_GRAPH_RERANK_ALPHA="${UCM_GRAPH_RERANK_ALPHA:-0.03}"
UCM_GRAPH_SOURCE_PRIOR_SAME_FILE="${UCM_GRAPH_SOURCE_PRIOR_SAME_FILE:-0.02}"
UCM_GRAPH_SOURCE_PRIOR_IDENTIFIER="${UCM_GRAPH_SOURCE_PRIOR_IDENTIFIER:-0.015}"
UCM_GRAPH_SOURCE_PRIOR_IMPORT="${UCM_GRAPH_SOURCE_PRIOR_IMPORT:-0.005}"
UCM_GRAPH_SOURCE_PRIOR_API_CALL="${UCM_GRAPH_SOURCE_PRIOR_API_CALL:-0.025}"
UCM_GRAPH_DISTANCE_PENALTY="${UCM_GRAPH_DISTANCE_PENALTY:-0.005}"

cd "${PROJECT_DIR}"
mkdir -p log_infer result_infer

if [[ -n "${UCM_RUN_TAG}" && "${UCM_RUN_TAG}" =~ [^A-Za-z0-9._-] ]]; then
  echo "Invalid UCM_RUN_TAG=${UCM_RUN_TAG}; use only letters, numbers, dot, underscore, or hyphen." >&2
  exit 2
fi
if [[ -n "${RUN_SHORT_NAME}" && "${RUN_SHORT_NAME}" =~ [^A-Za-z0-9._-] ]]; then
  echo "Invalid RUN_SHORT_NAME=${RUN_SHORT_NAME}; use only letters, numbers, dot, underscore, or hyphen." >&2
  exit 2
fi
if [[ ! "${GENERATOR_MAX_CONTEXT_LENGTH}" =~ ^[0-9]+$ || "${GENERATOR_MAX_CONTEXT_LENGTH}" -le 0 ]]; then
  echo "Invalid GENERATOR_MAX_CONTEXT_LENGTH=${GENERATOR_MAX_CONTEXT_LENGTH}; use a positive integer." >&2
  exit 2
fi
if [[ ! "${GENERATOR_MAX_CROSSFILE_LENGTH}" =~ ^[0-9]+$ || "${GENERATOR_MAX_CROSSFILE_LENGTH}" -le 0 ]]; then
  echo "Invalid GENERATOR_MAX_CROSSFILE_LENGTH=${GENERATOR_MAX_CROSSFILE_LENGTH}; use a positive integer." >&2
  exit 2
fi
if [[ "${GENERATOR_MAX_CROSSFILE_LENGTH}" -ge "${GENERATOR_MAX_CONTEXT_LENGTH}" ]]; then
  echo "GENERATOR_MAX_CROSSFILE_LENGTH must be smaller than GENERATOR_MAX_CONTEXT_LENGTH." >&2
  exit 2
fi

make_generator_window_name() {
  local infile_length=$((GENERATOR_MAX_CONTEXT_LENGTH - GENERATOR_MAX_CROSSFILE_LENGTH))
  printf "crossfile_%s_infile_%s" "${GENERATOR_MAX_CROSSFILE_LENGTH}" "${infile_length}"
}

make_output_dir() {
  local base_dir="$1"
  local include_ucm_suffix="${2:-0}"
  local suffix=""
  if [[ "${include_ucm_suffix}" == "1" ]]; then
    suffix="$(make_ucm_suffix)"
  fi
  if [[ -n "${UCM_RUN_TAG}" ]]; then
    suffix="${suffix:+${suffix}_}${UCM_RUN_TAG}"
  fi
  if [[ -n "${suffix}" ]]; then
    printf "%s_%s" "${base_dir}" "${suffix}"
  else
    printf "%s" "${base_dir}"
  fi
}

make_ucm_suffix() {
  local id_part="id"
  local import_part="import"
  local path_part="nopath"
  local bm25_part="enhbm25"
  local gate_part="nogate"
  local graph_part=""

  if [[ "${UCM_DISABLE_IDENTIFIER_QUERY}" == "1" ]]; then
    id_part="noid"
  fi
  if [[ "${UCM_DISABLE_IMPORT_API_QUERY}" == "1" ]]; then
    import_part="noimport"
  fi
  if [[ "${UCM_ENABLE_PATH_QUERY}" == "1" ]]; then
    path_part="path"
  fi
  if [[ "${UCM_DISABLE_ENHANCED_BM25}" == "1" ]]; then
    bm25_part="legacybm25"
  fi
  if [[ "${UCM_ENABLE_CONTEXT_GATE}" == "1" ]]; then
    gate_part="gate_aux${UCM_GATE_MAX_AUXILIARY_BLOCKS}_path${UCM_GATE_ALLOW_PATH_ONLY}_stop${UCM_GATE_STOP_RANK_THRESHOLD}"
  fi
  if [[ "${UCM_ENABLE_CONTEXT_GRAPH}" == "1" ]]; then
    graph_part="graph_s${UCM_GRAPH_MAX_SEED}_n${UCM_GRAPH_MAX_NEIGHBORS_PER_SEED}_e${UCM_GRAPH_MAX_EXPANDED}_scored"
    if [[ "${UCM_GRAPH_ENABLE_IDENTIFIER_EDGES}" == "1" ]]; then
      graph_part="${graph_part}_gid"
    fi
    if [[ "${UCM_GRAPH_ENABLE_IMPORT_EDGES}" == "1" ]]; then
      graph_part="${graph_part}_gimport"
    fi
    if [[ "${UCM_GRAPH_ENABLE_API_CALL_EDGES}" == "1" ]]; then
      graph_part="${graph_part}_gapi"
    fi
    if [[ "${UCM_GRAPH_ENABLE_RERANK}" == "1" ]]; then
      graph_part="${graph_part}_rerank_a${UCM_GRAPH_RERANK_ALPHA}_d${UCM_GRAPH_DISTANCE_PENALTY}"
    fi
  fi

  printf "%s_%s_%s_base%s_aux%s_pathk%s_pool%s_%s_%s%s" \
    "${id_part}" \
    "${import_part}" \
    "${path_part}" \
    "${UCM_BASE_TOPK}" \
    "${UCM_TOPK_PER_PATH}" \
    "${UCM_PATH_TOPK}" \
    "${UCM_CANDIDATE_POOL_SIZE}" \
    "${bm25_part}" \
    "${gate_part}" \
    "${graph_part:+_${graph_part}}"
}

export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Run mode: ${RUN_MODE}"
echo "Run tag: ${UCM_RUN_TAG:-default}"
echo "Run short name: ${RUN_SHORT_NAME:-default}"
echo "UCM source switches: disable_identifier=${UCM_DISABLE_IDENTIFIER_QUERY}, disable_import_api=${UCM_DISABLE_IMPORT_API_QUERY}, enable_path=${UCM_ENABLE_PATH_QUERY}"
echo "UCM enhanced BM25: disabled=${UCM_DISABLE_ENHANCED_BM25}"
echo "UCM context gate: enable=${UCM_ENABLE_CONTEXT_GATE}, max_aux=${UCM_GATE_MAX_AUXILIARY_BLOCKS}, allow_path_only=${UCM_GATE_ALLOW_PATH_ONLY}, stop_threshold=${UCM_GATE_STOP_RANK_THRESHOLD}"
echo "UCM context graph: enable=${UCM_ENABLE_CONTEXT_GRAPH}, max_seed=${UCM_GRAPH_MAX_SEED}, neighbors=${UCM_GRAPH_MAX_NEIGHBORS_PER_SEED}, expanded=${UCM_GRAPH_MAX_EXPANDED}, identifier_edges=${UCM_GRAPH_ENABLE_IDENTIFIER_EDGES}, import_edges=${UCM_GRAPH_ENABLE_IMPORT_EDGES}, api_call_edges=${UCM_GRAPH_ENABLE_API_CALL_EDGES}, same_file_direction=${UCM_GRAPH_SAME_FILE_DIRECTION}, identifier_query_only=${UCM_GRAPH_IDENTIFIER_QUERY_ONLY}, api_call_query_only=${UCM_GRAPH_API_CALL_QUERY_ONLY}"
echo "UCM graph scoring: seed_decay=${UCM_GRAPH_SEED_RANK_DECAY}, distance_decay=${UCM_GRAPH_DISTANCE_DECAY}, same_file_w=${UCM_GRAPH_SAME_FILE_WEIGHT}, identifier_w=${UCM_GRAPH_IDENTIFIER_WEIGHT}, import_w=${UCM_GRAPH_IMPORT_WEIGHT}, api_call_w=${UCM_GRAPH_API_CALL_WEIGHT}, query_bonus=${UCM_GRAPH_QUERY_OVERLAP_BONUS}, query_api_bonus=${UCM_GRAPH_QUERY_API_BONUS}"
echo "UCM graph rerank: enable=${UCM_GRAPH_ENABLE_RERANK}, alpha=${UCM_GRAPH_RERANK_ALPHA}, same_file_prior=${UCM_GRAPH_SOURCE_PRIOR_SAME_FILE}, identifier_prior=${UCM_GRAPH_SOURCE_PRIOR_IDENTIFIER}, import_prior=${UCM_GRAPH_SOURCE_PRIOR_IMPORT}, api_call_prior=${UCM_GRAPH_SOURCE_PRIOR_API_CALL}, distance_penalty=${UCM_GRAPH_DISTANCE_PENALTY}"
echo "Generator prompt window: context=${GENERATOR_MAX_CONTEXT_LENGTH}, crossfile=${GENERATOR_MAX_CROSSFILE_LENGTH}, infile=$((GENERATOR_MAX_CONTEXT_LENGTH - GENERATOR_MAX_CROSSFILE_LENGTH))"
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
  --generator_max_crossfile_length "${GENERATOR_MAX_CROSSFILE_LENGTH}"
  --generator_max_context_length "${GENERATOR_MAX_CONTEXT_LENGTH}"
  --generator_batch_size_per_gpu "${GENERATOR_BATCH_SIZE_PER_GPU}"
  --retriever_batch_size_per_gpu "${RETRIEVER_BATCH_SIZE_PER_GPU}"
  --num_workers "${NUM_WORKERS}"
)

if [[ -n "${RUN_SHORT_NAME}" ]]; then
  COMMON_ARGS+=(--run_short_name "${RUN_SHORT_NAME}")
fi

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

if [[ "${UCM_DISABLE_ENHANCED_BM25}" == "1" ]]; then
  UCM_ARGS+=(--ucm_disable_enhanced_bm25)
fi

if [[ "${UCM_DISABLE_IDENTIFIER_QUERY}" == "1" ]]; then
  UCM_ARGS+=(--ucm_disable_identifier_query)
fi

if [[ "${UCM_DISABLE_IMPORT_API_QUERY}" == "1" ]]; then
  UCM_ARGS+=(--ucm_disable_import_api_query)
fi

if [[ "${UCM_ENABLE_PATH_QUERY}" == "1" ]]; then
  UCM_ARGS+=(--ucm_enable_path_query)
fi

if [[ "${UCM_ENABLE_CONTEXT_GATE}" == "1" ]]; then
  UCM_ARGS+=(
    --enable_context_gate
    --ucm_gate_max_auxiliary_blocks "${UCM_GATE_MAX_AUXILIARY_BLOCKS}"
    --ucm_gate_allow_path_only "${UCM_GATE_ALLOW_PATH_ONLY}"
    --ucm_gate_stop_rank_threshold "${UCM_GATE_STOP_RANK_THRESHOLD}"
  )
fi

if [[ "${UCM_ENABLE_CONTEXT_GRAPH}" == "1" ]]; then
  UCM_ARGS+=(
    --enable_context_graph
    --ucm_graph_max_seed "${UCM_GRAPH_MAX_SEED}"
    --ucm_graph_max_neighbors_per_seed "${UCM_GRAPH_MAX_NEIGHBORS_PER_SEED}"
    --ucm_graph_max_expanded "${UCM_GRAPH_MAX_EXPANDED}"
    --ucm_graph_identifier_max_df "${UCM_GRAPH_IDENTIFIER_MAX_DF}"
    --ucm_graph_api_call_max_df "${UCM_GRAPH_API_CALL_MAX_DF}"
    --ucm_graph_same_file_direction "${UCM_GRAPH_SAME_FILE_DIRECTION}"
    --ucm_graph_seed_rank_decay "${UCM_GRAPH_SEED_RANK_DECAY}"
    --ucm_graph_distance_decay "${UCM_GRAPH_DISTANCE_DECAY}"
    --ucm_graph_same_file_weight "${UCM_GRAPH_SAME_FILE_WEIGHT}"
    --ucm_graph_identifier_weight "${UCM_GRAPH_IDENTIFIER_WEIGHT}"
    --ucm_graph_import_weight "${UCM_GRAPH_IMPORT_WEIGHT}"
    --ucm_graph_api_call_weight "${UCM_GRAPH_API_CALL_WEIGHT}"
    --ucm_graph_query_overlap_bonus "${UCM_GRAPH_QUERY_OVERLAP_BONUS}"
    --ucm_graph_query_api_bonus "${UCM_GRAPH_QUERY_API_BONUS}"
    --ucm_graph_rerank_alpha "${UCM_GRAPH_RERANK_ALPHA}"
    --ucm_graph_source_prior_same_file "${UCM_GRAPH_SOURCE_PRIOR_SAME_FILE}"
    --ucm_graph_source_prior_identifier "${UCM_GRAPH_SOURCE_PRIOR_IDENTIFIER}"
    --ucm_graph_source_prior_import "${UCM_GRAPH_SOURCE_PRIOR_IMPORT}"
    --ucm_graph_source_prior_api_call "${UCM_GRAPH_SOURCE_PRIOR_API_CALL}"
    --ucm_graph_distance_penalty "${UCM_GRAPH_DISTANCE_PENALTY}"
  )
fi

if [[ "${UCM_GRAPH_ENABLE_RERANK}" == "1" ]]; then
  UCM_ARGS+=(--ucm_graph_enable_rerank)
fi

if [[ "${UCM_GRAPH_IDENTIFIER_QUERY_ONLY}" == "1" ]]; then
  UCM_ARGS+=(--ucm_graph_identifier_query_only)
fi

if [[ "${UCM_GRAPH_API_CALL_QUERY_ONLY}" == "1" ]]; then
  UCM_ARGS+=(--ucm_graph_api_call_query_only)
fi

if [[ "${UCM_GRAPH_ENABLE_IDENTIFIER_EDGES}" == "1" ]]; then
  UCM_ARGS+=(--ucm_graph_enable_identifier_edges)
fi

if [[ "${UCM_GRAPH_ENABLE_IMPORT_EDGES}" == "1" ]]; then
  UCM_ARGS+=(--ucm_graph_enable_import_edges)
fi

if [[ "${UCM_GRAPH_ENABLE_API_CALL_EDGES}" == "1" ]]; then
  UCM_ARGS+=(--ucm_graph_enable_api_call_edges)
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
    OUTPUT_DIR="$(make_output_dir "result_infer/UCM_multipath_debug" 1)"
    echo "Output dir: ${OUTPUT_DIR}"
    conda run -p "${CONDA_ENV}" python main.py \
      "${COMMON_ARGS[@]}" \
      "${UCM_ARGS[@]}" \
      --debug \
      --ucm_trace_retrieval \
      --output_dir "${OUTPUT_DIR}"
    ;;

  ucm_repocoder_debug)
    OUTPUT_DIR="$(make_output_dir "result_infer/UCM_multipath_repocoder_debug" 1)"
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
    OUTPUT_DIR="$(make_output_dir "result_infer/UCM_multipath_deepseekcoder_7b_$(make_generator_window_name)" 1)"
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
