#!/bin/bash
set -euo pipefail

# Submit the G4 combined run and two typed-relation ablations.
#
# Usage:
#   bash scripts/submit_ucm_graph_g4_d4_ablation.sh [--dry-run] [d4|type-only|call-only ...]
#
# EVAL_DATASETS uses ':' between dataset names because Slurm reserves commas
# inside --export. The base job converts ':' back to ',' before invoking Python.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SLURM="${SCRIPT_DIR}/submit_ucm_a800.slurm"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ "$#" -eq 0 ]]; then
  CASES=(d4 type-only call-only)
else
  CASES=("$@")
fi

SUBMIT_EVAL_DATASETS="${EVAL_DATASETS:-cceval_java:repoeval_line}"
if [[ "${SUBMIT_EVAL_DATASETS}" == *,* ]]; then
  echo "EVAL_DATASETS must use ':' between names when submitted through Slurm." >&2
  echo "Example: EVAL_DATASETS=cceval_java:repoeval_line" >&2
  exit 2
fi

COMMON_EXPORTS=(
  "RUN_MODE=ucm_full"
  "EVAL_DATASETS=${SUBMIT_EVAL_DATASETS}"
  "GENERATOR_MAX_CONTEXT_LENGTH=4096"
  "GENERATOR_MAX_CROSSFILE_LENGTH=3072"
  "UCM_BASE_TOPK=60"
  "UCM_TOPK_PER_PATH=30"
  "UCM_PATH_TOPK=5"
  "UCM_CANDIDATE_POOL_SIZE=140"
  "UCM_DISABLE_IDENTIFIER_QUERY=0"
  "UCM_DISABLE_IMPORT_API_QUERY=0"
  "UCM_ENABLE_PATH_QUERY=0"
  "UCM_DISABLE_ENHANCED_BM25=0"
  "UCM_ENABLE_CONTEXT_GATE=0"
  "UCM_ENABLE_CONTEXT_GRAPH=1"
  "UCM_GRAPH_MAX_SEED=20"
  "UCM_GRAPH_MAX_NEIGHBORS_PER_SEED=2"
  "UCM_GRAPH_MAX_EXPANDED=20"
  "UCM_GRAPH_ENABLE_SAME_FILE_EDGES=0"
  "UCM_GRAPH_ENABLE_IDENTIFIER_EDGES=0"
  "UCM_GRAPH_ENABLE_IMPORT_EDGES=0"
  "UCM_GRAPH_ENABLE_API_CALL_EDGES=0"
  "UCM_GRAPH_ENABLE_TYPED_DEPENDENCY_EDGES=1"
  "UCM_GRAPH_TYPED_RELATION_MODE=all"
  "UCM_GRAPH_TYPED_QUERY_MAX=8"
  "UCM_GRAPH_TYPED_QUERY_CONTEXT_LINES=80"
  "UCM_GRAPH_TYPED_MAX_DF=12"
  "UCM_GRAPH_TYPED_QUERY_BONUS=1.0"
  "UCM_GRAPH_TYPED_CALL_WEIGHT=1.8"
  "UCM_GRAPH_TYPED_TYPE_WEIGHT=2.0"
  "UCM_GRAPH_TYPED_DEF_USE_WEIGHT=1.4"
  "UCM_GRAPH_TYPED_ALLOW_TARGET_FILE=0"
  "UCM_GRAPH_MAX_SELECTED=2"
  "UCM_GRAPH_MAX_SELECTED_TOKENS=768"
  "UCM_GRAPH_RELATION_MAX_SYMBOLS=3"
  "UCM_GRAPH_SHOW_RELATIONS_IN_PROMPT=1"
  "UCM_GRAPH_ENABLE_RERANK=1"
  "UCM_GRAPH_RERANK_ALPHA=0.03"
  "UCM_GRAPH_SOURCE_PRIOR_SAME_FILE=0"
  "UCM_GRAPH_SOURCE_PRIOR_IDENTIFIER=0"
  "UCM_GRAPH_SOURCE_PRIOR_IMPORT=0"
  "UCM_GRAPH_SOURCE_PRIOR_API_CALL=0"
  "UCM_GRAPH_DISTANCE_PENALTY=0"
)

join_by_comma() {
  local IFS=","
  echo "$*"
}

submit_case() {
  local short_name="$1"
  shift
  declare -A env_map=()
  local item key value
  for item in "${COMMON_EXPORTS[@]}" "$@" "RUN_SHORT_NAME=${short_name}"; do
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
  echo "  datasets=${env_map[EVAL_DATASETS]} relation_mode=${env_map[UCM_GRAPH_TYPED_RELATION_MODE]} visible=${env_map[UCM_GRAPH_SHOW_RELATIONS_IN_PROMPT]} rerank=${env_map[UCM_GRAPH_ENABLE_RERANK]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  sbatch --export=ALL,<experiment-config> ${BASE_SLURM}"
  else
    sbatch --export="ALL,${exports}" "${BASE_SLURM}"
  fi
}

for case_name in "${CASES[@]}"; do
  case "${case_name}" in
    d4)
      submit_case "G4_D4_ddg_lite_visible_rerank_java_line_4k"
      ;;
    type-only)
      submit_case "G4_D5_ddg_lite_type_only_visible_rerank_java_line_4k" \
        "UCM_GRAPH_TYPED_RELATION_MODE=type_only"
      ;;
    call-only)
      submit_case "G4_D6_ddg_lite_call_only_visible_rerank_java_line_4k" \
        "UCM_GRAPH_TYPED_RELATION_MODE=call_only"
      ;;
    *)
      echo "Unknown case: ${case_name}" >&2
      echo "Choose from: d4, type-only, call-only" >&2
      exit 2
      ;;
  esac
done
