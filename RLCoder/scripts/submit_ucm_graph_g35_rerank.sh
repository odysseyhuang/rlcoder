#!/bin/bash
set -euo pipefail

# G3.5 graph-aware rerank ablations on the fixed UCM graph base.
#
# Usage:
#   bash scripts/submit_ucm_graph_g35_rerank.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SLURM="${SCRIPT_DIR}/submit_ucm_a800.slurm"

COMMON_EXPORTS=(
  "RUN_MODE=ucm_full"
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
  "UCM_GRAPH_ENABLE_RERANK=0"
  "UCM_GRAPH_RERANK_ALPHA=0.03"
  "UCM_GRAPH_SOURCE_PRIOR_SAME_FILE=0.02"
  "UCM_GRAPH_SOURCE_PRIOR_IDENTIFIER=0.015"
  "UCM_GRAPH_SOURCE_PRIOR_IMPORT=0.005"
  "UCM_GRAPH_SOURCE_PRIOR_API_CALL=0.025"
  "UCM_GRAPH_DISTANCE_PENALTY=0.005"
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

submit_case "G3.5_B0_a1_repro_4k" \
  "UCM_GRAPH_ENABLE_RERANK=0"

submit_case "G3.5_R1_graphscore_4k" \
  "UCM_GRAPH_ENABLE_RERANK=1" \
  "UCM_GRAPH_RERANK_ALPHA=0.03" \
  "UCM_GRAPH_SOURCE_PRIOR_SAME_FILE=0" \
  "UCM_GRAPH_SOURCE_PRIOR_IDENTIFIER=0" \
  "UCM_GRAPH_SOURCE_PRIOR_IMPORT=0" \
  "UCM_GRAPH_SOURCE_PRIOR_API_CALL=0" \
  "UCM_GRAPH_DISTANCE_PENALTY=0"

submit_case "G3.5_R2_sourceprior_4k" \
  "UCM_GRAPH_ENABLE_RERANK=1" \
  "UCM_GRAPH_RERANK_ALPHA=0" \
  "UCM_GRAPH_SOURCE_PRIOR_SAME_FILE=0.02" \
  "UCM_GRAPH_SOURCE_PRIOR_IDENTIFIER=0.015" \
  "UCM_GRAPH_SOURCE_PRIOR_IMPORT=0.005" \
  "UCM_GRAPH_SOURCE_PRIOR_API_CALL=0.025" \
  "UCM_GRAPH_DISTANCE_PENALTY=0"

submit_case "G3.5_R3_score_source_4k" \
  "UCM_GRAPH_ENABLE_RERANK=1"

submit_case "G3.5_R4_score_source_dist_4k" \
  "UCM_GRAPH_ENABLE_RERANK=1" \
  "UCM_GRAPH_DISTANCE_PENALTY=0.005"

submit_case "G3.5_A3_R3_api_score_source_4k" \
  "UCM_GRAPH_ENABLE_RERANK=1" \
  "UCM_GRAPH_ENABLE_API_CALL_EDGES=1"
