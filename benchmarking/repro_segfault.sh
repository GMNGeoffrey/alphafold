#!/usr/bin/env bash

# Repro issues with jax disk caching

set -euo pipefail

# Required environment variables:
#   INPUT_DIR: Directory containing input feature .npz files.
#   DATA_DIR: Directory containing Alphafold model parameter files under params/ subdir.

MODEL_INDEX=1
COMPLEX_NAME="7u8c"

OUTPUT_DIR="${OUTPUT_DIR:-output/repro/$(date +%F_%H-%M-%S)}"
echo "Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

(
  set -x;
  python run_alphafold_model_only.py \
    --data_dir="${DATA_DIR}" \
    --feature_paths="${INPUT_DIR}/${COMPLEX_NAME}.npz" \
    --model_preset=multimer \
    --models_to_relax=none \
    --model_names="model_1_multimer_v3" \
    --num_multimer_predictions_per_model=1 \
    --output_dir="${OUTPUT_DIR}" \
    --random_seed=1 \
    --consistent_random_seeds \
    --use_gpu_relax=False \
    --num_recycle=1 \
    --recycle_early_stop_tolerance=-1 \
) 2>&1 | tee -a "${OUTPUT_DIR}/output.log"
