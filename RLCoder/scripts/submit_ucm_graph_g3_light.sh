#!/bin/bash
set -euo pipefail

# G3-UCG-Light ablations on the fixed UCM base:
# id+import query views, no path query, base60/aux30/pool140, enhanced BM25.
#
# Usage:
#   bash scripts/submit_ucm_graph_g3_light.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SLURM="${SCRIPT_DIR}/submit_ucm_a800.slurm"

COMMON_EXPORTS=(
  "RUN_MODE=ucm_full"
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
  "UCM_GRAPH_MAX_EXPANDED=40"
  "UCM_GRAPH_SAME_FILE_DIRECTION=both"
  "UCM_GRAPH_ENABLE_IDENTIFIER_EDGES=1"
  "UCM_GRAPH_ENABLE_IMPORT_EDGES=0"
  "UCM_GRAPH_ENABLE_API_CALL_EDGES=0"
  "UCM_GRAPH_IDENTIFIER_MAX_DF=10"
  "UCM_GRAPH_API_CALL_MAX_DF=20"
  "UCM_GRAPH_IDENTIFIER_QUERY_ONLY=1"
  "UCM_GRAPH_API_CALL_QUERY_ONLY=0"
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
  sbatch --export="ALL,${exports}" "${BASE_SLURM}"
}

submit_case "G3_A0_base_ucm" \
  "UCM_ENABLE_CONTEXT_GRAPH=0" \
  "UCM_GRAPH_ENABLE_IDENTIFIER_EDGES=0"

submit_case "G3_A1_g2_best_like" \
  "UCM_ENABLE_CONTEXT_GRAPH=1"

submit_case "G3_A2_import_path" \
  "UCM_ENABLE_CONTEXT_GRAPH=1" \
  "UCM_GRAPH_ENABLE_IMPORT_EDGES=1"

submit_case "G3_A3_api_call" \
  "UCM_ENABLE_CONTEXT_GRAPH=1" \
  "UCM_GRAPH_ENABLE_API_CALL_EDGES=1"

submit_case "G3_A4_full_light" \
  "UCM_ENABLE_CONTEXT_GRAPH=1" \
  "UCM_GRAPH_ENABLE_IMPORT_EDGES=1" \
  "UCM_GRAPH_ENABLE_API_CALL_EDGES=1"

submit_case "G3_A5_full_light_gate_or_weighted" \
  "UCM_ENABLE_CONTEXT_GRAPH=1" \
  "UCM_GRAPH_ENABLE_IMPORT_EDGES=1" \
  "UCM_GRAPH_ENABLE_API_CALL_EDGES=1" \
  "UCM_GRAPH_MAX_NEIGHBORS_PER_SEED=1" \
  "UCM_GRAPH_MAX_EXPANDED=30" \
  "UCM_GRAPH_IDENTIFIER_WEIGHT=1.0" \
  "UCM_GRAPH_IMPORT_WEIGHT=1.6" \
  "UCM_GRAPH_API_CALL_WEIGHT=1.8" \
  "UCM_GRAPH_API_CALL_QUERY_ONLY=1"
