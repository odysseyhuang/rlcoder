#!/bin/bash
set -euo pipefail

# Submit lightweight graph ablations on top of the fixed UCM base:
# id+import, no path query, base60/aux30/pool140, enhanced BM25, no gate.
#
# Usage:
#   bash scripts/submit_ucm_graph_lite_ablation.sh
#
# Optional overrides:
#   PROJECT_DIR=/path/to/RLCoder CONDA_ENV=/path/to/env bash scripts/submit_ucm_graph_lite_ablation.sh

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
  "UCM_ENABLE_CONTEXT_GRAPH=1"
  "UCM_GRAPH_MAX_SEED=20"
  "UCM_GRAPH_SEED_RANK_DECAY=0.05"
  "UCM_GRAPH_DISTANCE_DECAY=0.75"
  "UCM_GRAPH_SAME_FILE_WEIGHT=1.0"
  "UCM_GRAPH_IDENTIFIER_WEIGHT=1.2"
  "UCM_GRAPH_IMPORT_WEIGHT=1.4"
  "UCM_GRAPH_QUERY_OVERLAP_BONUS=2"
)

join_by_comma() {
  local IFS=","
  echo "$*"
}

submit_case() {
  local tag="$1"
  shift
  local exports
  exports="$(join_by_comma "${COMMON_EXPORTS[@]}" "$@" "UCM_RUN_TAG=${tag}")"
  echo "Submitting ${tag}"
  sbatch --export="ALL,${exports}" "${BASE_SLURM}"
}

submit_case "A2_graph_samefile_scored" \
  "UCM_GRAPH_MAX_NEIGHBORS_PER_SEED=1" \
  "UCM_GRAPH_MAX_EXPANDED=20" \
  "UCM_GRAPH_ENABLE_IDENTIFIER_EDGES=0" \
  "UCM_GRAPH_ENABLE_IMPORT_EDGES=0"

submit_case "A3_graph_samefile_identifier_scored" \
  "UCM_GRAPH_MAX_NEIGHBORS_PER_SEED=1" \
  "UCM_GRAPH_MAX_EXPANDED=30" \
  "UCM_GRAPH_ENABLE_IDENTIFIER_EDGES=1" \
  "UCM_GRAPH_ENABLE_IMPORT_EDGES=0"

submit_case "A4_graph_samefile_identifier_import_scored" \
  "UCM_GRAPH_MAX_NEIGHBORS_PER_SEED=1" \
  "UCM_GRAPH_MAX_EXPANDED=40" \
  "UCM_GRAPH_ENABLE_IDENTIFIER_EDGES=1" \
  "UCM_GRAPH_ENABLE_IMPORT_EDGES=1"
