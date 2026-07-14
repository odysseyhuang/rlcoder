#!/bin/bash
set -euo pipefail

# Second-round lightweight graph ablations.
# All cases use the fixed UCM base:
# base + identifier + import/API queries, no path query, enhanced BM25,
# base60/aux30/pool140, no gate.
#
# Usage:
#   bash scripts/submit_ucm_graph_g2_ablation.sh

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
  "UCM_GRAPH_IDENTIFIER_MAX_DF=20"
  "UCM_GRAPH_IDENTIFIER_QUERY_ONLY=0"
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

submit_case "G2_A1_base" \
  "UCM_ENABLE_CONTEXT_GRAPH=0" \
  "UCM_GRAPH_ENABLE_IDENTIFIER_EDGES=0"

submit_case "G2_A3_sf_id" \
  "UCM_ENABLE_CONTEXT_GRAPH=1"

submit_case "G2_S1_sfprev_id" \
  "UCM_ENABLE_CONTEXT_GRAPH=1" \
  "UCM_GRAPH_SAME_FILE_DIRECTION=prev"

submit_case "G2_S2_sfnext_id" \
  "UCM_ENABLE_CONTEXT_GRAPH=1" \
  "UCM_GRAPH_SAME_FILE_DIRECTION=next"

submit_case "G2_S3_sf1_id" \
  "UCM_ENABLE_CONTEXT_GRAPH=1" \
  "UCM_GRAPH_MAX_NEIGHBORS_PER_SEED=1" \
  "UCM_GRAPH_MAX_EXPANDED=30"

submit_case "G2_I1_sf_id_qonly" \
  "UCM_ENABLE_CONTEXT_GRAPH=1" \
  "UCM_GRAPH_IDENTIFIER_QUERY_ONLY=1"

submit_case "G2_I2_sf_id_qonly_df10" \
  "UCM_ENABLE_CONTEXT_GRAPH=1" \
  "UCM_GRAPH_IDENTIFIER_QUERY_ONLY=1" \
  "UCM_GRAPH_IDENTIFIER_MAX_DF=10"
