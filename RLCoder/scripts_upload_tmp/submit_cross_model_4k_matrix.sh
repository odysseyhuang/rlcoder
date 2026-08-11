#!/bin/bash
set -euo pipefail

# Cross-model 4K generalization matrix.
#
# Default: submit M0/M1/M2 for the three fixed 7B backbones:
#   DeepSeekCoder-6.7B-base, CodeLlama-7B-hf, StarCoderBase-7B.
#
# Usage:
#   bash scripts/submit_cross_model_4k_matrix.sh [--dry-run] [--include-m3] [--m0-m2-only] [--include-qwen]
#
# Common overrides:
#   PROJECT_DIR=/path/to/RLCoder CONDA_ENV=/path/to/env bash scripts/submit_cross_model_4k_matrix.sh
#   EVAL_DATASETS=cceval_python:cceval_java:repoeval_line:repoeval_api bash scripts/submit_cross_model_4k_matrix.sh
#   GENERATOR_BATCH_SIZE_PER_GPU=4 bash scripts/submit_cross_model_4k_matrix.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SLURM="${SCRIPT_DIR}/submit_ucm_a800.slurm"

DRY_RUN=0
INCLUDE_M3=0
M0_M2_ONLY=0
INCLUDE_QWEN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --include-m3)
      INCLUDE_M3=1
      ;;
    --m0-m2-only)
      M0_M2_ONLY=1
      ;;
    --include-qwen)
      INCLUDE_QWEN=1
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: bash scripts/submit_cross_model_4k_matrix.sh [--dry-run] [--include-m3] [--m0-m2-only] [--include-qwen]" >&2
      exit 2
      ;;
  esac
  shift
done

SUBMIT_EVAL_DATASETS="${EVAL_DATASETS:-cceval_python:cceval_java:repoeval_line:repoeval_api}"
if [[ "${SUBMIT_EVAL_DATASETS}" == *,* ]]; then
  echo "EVAL_DATASETS must use ':' between names when submitted through Slurm." >&2
  echo "Example: EVAL_DATASETS=cceval_python:cceval_java:repoeval_line:repoeval_api" >&2
  exit 2
fi

MODELS=(
  "deepseekcoder_7b|./models/deepseek-coder-6.7b-base"
  "codellama_7b|./models/CodeLlama-7b-hf"
  "starcoderbase_7b|./models/starcoderbase-7b"
)

if [[ "${INCLUDE_QWEN}" == "1" ]]; then
  MODELS+=("qwen25coder_7b|${QWEN_GENERATOR_MODEL:-./models/Qwen2.5-Coder-7B}")
fi

COMMON_EXPORTS=(
  "EVAL_DATASETS=${SUBMIT_EVAL_DATASETS}"
  "GENERATOR_MAX_CONTEXT_LENGTH=4096"
  "GENERATOR_MAX_CROSSFILE_LENGTH=3072"
  "GENERATOR_MAX_GENERATION_LENGTH=64"
  "RETRIEVER_QUERY_CONTEXT_LENGTH=256"
  "RETRIEVER_CANDIDATE_CONTEXT_LENGTH=512"
  "SAMPLE_NUMBER=10"
  "UCM_BASE_TOPK=60"
  "UCM_TOPK_PER_PATH=30"
  "UCM_PATH_TOPK=5"
  "UCM_CANDIDATE_POOL_SIZE=140"
  "UCM_DISABLE_IDENTIFIER_QUERY=0"
  "UCM_DISABLE_IMPORT_API_QUERY=0"
  "UCM_ENABLE_PATH_QUERY=0"
  "UCM_DISABLE_ENHANCED_BM25=0"
  "UCM_ENABLE_CONTEXT_GATE=0"
  "UCM_GRAPH_MAX_SEED=20"
  "UCM_GRAPH_MAX_NEIGHBORS_PER_SEED=2"
  "UCM_GRAPH_SAME_FILE_DIRECTION=both"
  "UCM_GRAPH_IDENTIFIER_MAX_DF=10"
  "UCM_GRAPH_API_CALL_MAX_DF=20"
  "UCM_GRAPH_SEED_RANK_DECAY=0.05"
  "UCM_GRAPH_DISTANCE_DECAY=0.75"
  "UCM_GRAPH_SAME_FILE_WEIGHT=1.0"
  "UCM_GRAPH_IDENTIFIER_WEIGHT=1.2"
  "UCM_GRAPH_IMPORT_WEIGHT=1.4"
  "UCM_GRAPH_API_CALL_WEIGHT=1.6"
  "UCM_GRAPH_QUERY_OVERLAP_BONUS=2"
  "UCM_GRAPH_QUERY_API_BONUS=2"
)

join_by_comma() {
  local IFS=","
  echo "$*"
}

submit_case() {
  local model_alias="$1"
  local generator_model="$2"
  local short_name="$3"
  shift 3

  declare -A env_map=()
  local item key value
  for item in "${COMMON_EXPORTS[@]}" \
      "GENERATOR_ALIAS=${model_alias}" \
      "GENERATOR_MODEL=${generator_model}" \
      "RUN_SHORT_NAME=${short_name}" \
      "$@"; do
    key="${item%%=*}"
    value="${item#*=}"
    env_map["${key}"]="${value}"
  done

  local export_items=()
  for key in "${!env_map[@]}"; do
    export_items+=("${key}=${env_map[$key]}")
  done

  local exports
  exports="$(join_by_comma "${export_items[@]}")"
  echo "Submitting ${short_name}"
  echo "  model=${model_alias} path=${generator_model} mode=${env_map[RUN_MODE]} datasets=${env_map[EVAL_DATASETS]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  sbatch --export=ALL,${exports} ${BASE_SLURM}"
  else
    sbatch --export="ALL,${exports}" "${BASE_SLURM}"
  fi
}

submit_model_matrix() {
  local model_alias="$1"
  local generator_model="$2"

  submit_case "${model_alias}" "${generator_model}" "X4K_${model_alias}_M0_rlcoder" \
    "RUN_MODE=rlcoder_full"

  if [[ "${M0_M2_ONLY}" != "1" ]]; then
    submit_case "${model_alias}" "${generator_model}" "X4K_${model_alias}_M1_ucm_nograph" \
      "RUN_MODE=ucm_full" \
      "UCM_ENABLE_CONTEXT_GRAPH=0" \
      "UCM_GRAPH_ENABLE_SAME_FILE_EDGES=0" \
      "UCM_GRAPH_ENABLE_IDENTIFIER_EDGES=0" \
      "UCM_GRAPH_ENABLE_IMPORT_EDGES=0" \
      "UCM_GRAPH_ENABLE_API_CALL_EDGES=0" \
      "UCM_GRAPH_ENABLE_TYPED_DEPENDENCY_EDGES=0" \
      "UCM_GRAPH_MAX_EXPANDED=0" \
      "UCM_GRAPH_MAX_SELECTED=0" \
      "UCM_GRAPH_MAX_SELECTED_TOKENS=0" \
      "UCM_GRAPH_SHOW_RELATIONS_IN_PROMPT=0" \
      "UCM_GRAPH_ENABLE_RERANK=0"
  fi

  submit_case "${model_alias}" "${generator_model}" "X4K_${model_alias}_M2_ucm_graph_light_best" \
    "RUN_MODE=ucm_full" \
    "UCM_ENABLE_CONTEXT_GRAPH=1" \
    "UCM_GRAPH_MAX_EXPANDED=40" \
    "UCM_GRAPH_ENABLE_SAME_FILE_EDGES=1" \
    "UCM_GRAPH_ENABLE_IDENTIFIER_EDGES=1" \
    "UCM_GRAPH_ENABLE_IMPORT_EDGES=1" \
    "UCM_GRAPH_ENABLE_API_CALL_EDGES=1" \
    "UCM_GRAPH_ENABLE_TYPED_DEPENDENCY_EDGES=0" \
    "UCM_GRAPH_IDENTIFIER_QUERY_ONLY=1" \
    "UCM_GRAPH_API_CALL_QUERY_ONLY=0" \
    "UCM_GRAPH_ENABLE_RERANK=0" \
    "UCM_GRAPH_MAX_SELECTED=0" \
    "UCM_GRAPH_MAX_SELECTED_TOKENS=0" \
    "UCM_GRAPH_SHOW_RELATIONS_IN_PROMPT=0"

  if [[ "${INCLUDE_M3}" == "1" && "${M0_M2_ONLY}" != "1" ]]; then
    submit_case "${model_alias}" "${generator_model}" "X4K_${model_alias}_M3_g4d3_typed_rerank" \
      "RUN_MODE=ucm_full" \
      "UCM_ENABLE_CONTEXT_GRAPH=1" \
      "UCM_GRAPH_MAX_EXPANDED=20" \
      "UCM_GRAPH_ENABLE_SAME_FILE_EDGES=0" \
      "UCM_GRAPH_ENABLE_IDENTIFIER_EDGES=0" \
      "UCM_GRAPH_ENABLE_IMPORT_EDGES=0" \
      "UCM_GRAPH_ENABLE_API_CALL_EDGES=0" \
      "UCM_GRAPH_ENABLE_TYPED_DEPENDENCY_EDGES=1" \
      "UCM_GRAPH_TYPED_RELATION_MODE=all" \
      "UCM_GRAPH_TYPED_QUERY_MAX=8" \
      "UCM_GRAPH_TYPED_QUERY_CONTEXT_LINES=80" \
      "UCM_GRAPH_TYPED_MAX_DF=12" \
      "UCM_GRAPH_TYPED_QUERY_BONUS=1.0" \
      "UCM_GRAPH_TYPED_CALL_WEIGHT=1.8" \
      "UCM_GRAPH_TYPED_TYPE_WEIGHT=2.0" \
      "UCM_GRAPH_TYPED_DEF_USE_WEIGHT=1.4" \
      "UCM_GRAPH_TYPED_ALLOW_TARGET_FILE=0" \
      "UCM_GRAPH_MAX_SELECTED=2" \
      "UCM_GRAPH_MAX_SELECTED_TOKENS=768" \
      "UCM_GRAPH_RELATION_MAX_SYMBOLS=3" \
      "UCM_GRAPH_SHOW_RELATIONS_IN_PROMPT=0" \
      "UCM_GRAPH_ENABLE_RERANK=1" \
      "UCM_GRAPH_RERANK_ALPHA=0.03" \
      "UCM_GRAPH_SOURCE_PRIOR_SAME_FILE=0" \
      "UCM_GRAPH_SOURCE_PRIOR_IDENTIFIER=0" \
      "UCM_GRAPH_SOURCE_PRIOR_IMPORT=0" \
      "UCM_GRAPH_SOURCE_PRIOR_API_CALL=0" \
      "UCM_GRAPH_DISTANCE_PENALTY=0"
  fi
}

for model_spec in "${MODELS[@]}"; do
  model_alias="${model_spec%%|*}"
  generator_model="${model_spec#*|}"
  submit_model_matrix "${model_alias}" "${generator_model}"
done
