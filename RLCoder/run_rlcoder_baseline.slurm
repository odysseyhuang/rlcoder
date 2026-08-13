#!/bin/bash
#SBATCH --job-name=rlcoder_baseline
#SBATCH --partition=gpuA800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=log_infer/%j_baseline.out
#SBATCH --error=log_infer/%j_baseline.err

set -e

cd /share/home/202421045349/projects/RLCoder/RLCoder

mkdir -p log_infer result_infer/RLCoder_deepseekcoder_7b_crossfile_3072_infile_1024

export TOKENIZERS_PARALLELISM=false

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

conda run -p /share/home/202421045349/conda_envs/rlcoder python main.py \
  --eval \
  --weighted_keywords \
  --enable_generation \
  --inference_type unixcoder_with_rl \
  --retriever_model_path ./models/RLRetriever \
  --generator_model_path ./models/deepseek-coder-6.7b-base \
  --generator_max_crossfile_length 3072 \
  --generator_max_context_length 4096 \
  --generator_batch_size_per_gpu 8 \
  --retriever_batch_size_per_gpu 32 \
  --num_workers 6 \
  --output_dir result_infer/RLCoder_deepseekcoder_7b_crossfile_3072_infile_1024

echo "End time: $(date)"
