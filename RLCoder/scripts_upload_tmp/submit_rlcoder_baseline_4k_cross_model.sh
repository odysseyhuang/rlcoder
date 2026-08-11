#!/bin/bash
set -euo pipefail

# Submit bare RLCoder 4K baselines for locally deployed 7B generators other
# than the existing DeepSeekCoder run. All settings match
# RLCoder_deepseekcoder_7b_crossfile_3072_infile_1024 except generator model.
#
# Usage:
#   bash scripts/submit_rlcoder_baseline_4k_cross_model.sh [--dry-run] [codellama|starcoderbase ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SLURM="${SCRIPT_DIR}/submit_ucm_a800.slurm"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ "$#" -eq 0 ]]; then
  CASES=(codellama starcoderbase)
else
  CASES=("$@")
fi

join_by_comma() {
  local IFS=","
  echo "$*"
}

submit_case() {
  local alias="$1"
  local model_path="$2"
  local short_name="RLCoder_${alias}_crossfile_3072_infile_1024"
  local exports

  exports="$(join_by_comma \
    "RUN_MODE=rlcoder_full" \
    "RUN_SHORT_NAME=${short_name}" \
    "GENERATOR_ALIAS=${alias}" \
    "GENERATOR_MODEL=${model_path}" \
    "GENERATOR_MAX_CONTEXT_LENGTH=4096" \
    "GENERATOR_MAX_CROSSFILE_LENGTH=3072" \
    "GENERATOR_MAX_GENERATION_LENGTH=64" \
    "RETRIEVER_QUERY_CONTEXT_LENGTH=256" \
    "RETRIEVER_CANDIDATE_CONTEXT_LENGTH=512" \
    "SAMPLE_NUMBER=10")"

  echo "Submitting ${short_name}"
  echo "  generator=${model_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  sbatch --export=ALL,${exports} ${BASE_SLURM}"
  else
    sbatch --export="ALL,${exports}" "${BASE_SLURM}"
  fi
}

for case_name in "${CASES[@]}"; do
  case "${case_name}" in
    codellama)
      submit_case "codellama_7b" "./models/CodeLlama-7b-hf"
      ;;
    starcoderbase)
      submit_case "starcoderbase_7b" "./models/starcoderbase-7b"
      ;;
    *)
      echo "Unknown case: ${case_name}" >&2
      echo "Choose from: codellama, starcoderbase" >&2
      exit 2
      ;;
  esac
done
