#!/usr/bin/env bash

# Repro issues with jax disk caching

set -euo pipefail

# Required environment variables:
# Variable per job:
#   MODEL_INDEX: Index of the Alphafold model to run (1-5 for multimer_v3).
# The same for all jobs in a run
#   INPUT_DIR: Directory containing input feature .npz files.
#   DATA_DIR: Directory containing Alphafold model parameter files under params/ subdir.
#   OUTPUT_DIR: Root directory to write output results to.

GPU=0
SEED=1
COMPLEX_NAME=7u8c

MODEL_NAME="model_${MODEL_INDEX}_multimer_v3"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${COMPLEX_NAME}/seed_${SEED}/model_${MODEL_INDEX}"

mkdir -p "${RUN_OUTPUT_DIR}"


export JAX_COMPILATION_CACHE_DIR="$(mktemp -d --tmpdir jax_cache.XXX)"

echo "Using cache directory: ${JAX_COMPILATION_CACHE_DIR}" | tee -a "${RUN_OUTPUT_DIR}/output.log"

echo "Running with empty cache" | tee -a "${RUN_OUTPUT_DIR}/output.log"
(
  set -x;
  CUDA_VISIBLE_DEVICES="${GPU}" \
      python run_alphafold_model_only.py \
        --data_dir="${DATA_DIR}" \
        --feature_paths="${INPUT_DIR}/${COMPLEX_NAME}.npz" \
        --model_preset=multimer \
        --models_to_relax=none \
        --model_names="${MODEL_NAME}" \
        --num_multimer_predictions_per_model=1 \
        --output_dir="${RUN_OUTPUT_DIR}" \
        --random_seed="${SEED}" \
        --consistent_random_seeds \
        --use_gpu_relax \
        --num_recycle=1 \
        --recycle_early_stop_tolerance=-1 \
) 2>&1 | tee -a "${RUN_OUTPUT_DIR}/output.log"

echo "Running again with warm cache in new process" | tee -a "${RUN_OUTPUT_DIR}/output.log"

(
  set -x;
  CUDA_VISIBLE_DEVICES="${GPU}" \
      python run_alphafold_model_only.py \
        --data_dir="${DATA_DIR}" \
        --feature_paths="${INPUT_DIR}/${COMPLEX_NAME}.npz" \
        --model_preset=multimer \
        --models_to_relax=none \
        --model_names="${MODEL_NAME}" \
        --num_multimer_predictions_per_model=1 \
        --output_dir="${RUN_OUTPUT_DIR}" \
        --random_seed="${SEED}" \
        --consistent_random_seeds \
        --use_gpu_relax \
        --num_recycle=1 \
        --recycle_early_stop_tolerance=-1 \
) 2>&1 | tee -a "${RUN_OUTPUT_DIR}/output.log"
