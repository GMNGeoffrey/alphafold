#!/usr/bin/env bash

# Repro issues with jax disk caching

set -euo pipefail

# Required environment variables:
# Variable per job:
#   MODEL_INDEX: Index of the Alphafold model to run (1-5 for multimer_v3).
#   COMPLEX_NAME: Name of the protein complex to run (e.g. 7u8c or 7fci).
# The same for all jobs in a run
#   INPUT_DIR: Directory containing input feature .npz files.
#   DATA_DIR: Directory containing Alphafold model parameter files under params/ subdir.
#   OUTPUT_DIR: Root directory to write output results to.

SEED=1

PROFILE="${PROFILE:-0}"

MODEL_NAME="model_${MODEL_INDEX}_multimer_v3"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${COMPLEX_NAME}/seed_${SEED}/model_${MODEL_INDEX}"

echo "Creating output directory: ${RUN_OUTPUT_DIR}"
mkdir -p "${RUN_OUTPUT_DIR}"

if [[ -v JAX_COMPILATION_CACHE_DIR ]]; then
  echo "Using cache directory: ${JAX_COMPILATION_CACHE_DIR}" | tee -a "${RUN_OUTPUT_DIR}/output.log"
else
  echo "No JAX_COMPILATION_CACHE_DIR set" | tee -a "${RUN_OUTPUT_DIR}/output.log"
fi

(
  set -x;
  python run_alphafold_model_only.py \
    --benchmark \
    --profile="${PROFILE}" \
    --data_dir="${DATA_DIR}" \
    --feature_paths="${INPUT_DIR}/${COMPLEX_NAME}.npz" \
    --model_preset=multimer \
    --models_to_relax=none \
    --model_names="${MODEL_NAME}" \
    --num_multimer_predictions_per_model=1 \
    --output_dir="${RUN_OUTPUT_DIR}" \
    --random_seed="${SEED}" \
    --consistent_random_seeds \
    --use_gpu_relax=False \
    --num_recycle=1 \
    --recycle_early_stop_tolerance=-1 \
) 2>&1 | tee -a "${RUN_OUTPUT_DIR}/output.log"
