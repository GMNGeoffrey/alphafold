#!/usr/bin/env bash

set -euo pipefail

# Required environment variables:
# Variable per job:
#   COMPLEX_NAME: Name of the protein complex to benchmark (e.g., "7fci").
#   MODEL_INDEX: Index of the Alphafold model to run (1-5 for multimer_v3).
#   SEED: Random seed to use for this job.
#   SLOT: The parallel slot number (1 to TOTAL_GPUS) assigned by GNU parallel.
# The same for all jobs in a run
#   INPUT_DIR: Directory containing input feature .npz files.
#   DATA_DIR: Directory containing Alphafold model parameter files under params/ subdir.
#   OUTPUT_DIR: Root directory to write output results to.
#   CORES_PER_JOB: Number of CPU cores allocated per job (should be nproc // TOTAL_GPUS).

GPU="$((SLOT - 1))"
CPU_START="$(((SLOT - 1) * CORES_PER_JOB))"
CPU_END="$((CPU_START + CORES_PER_JOB - 1))"

MODEL_NAME="model_${MODEL_INDEX}_multimer_v3"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${COMPLEX_NAME}/seed_${SEED}/model_${MODEL_INDEX}"

mkdir -p "${RUN_OUTPUT_DIR}"

(
    set -x;
  CUDA_VISIBLE_DEVICES="${GPU}" taskset -c "${CPU_START}-${CPU_END}" \
       python run_alphafold_model_only.py \
         --benchmark \
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
) 2>&1 | tee "${RUN_OUTPUT_DIR}/output.log"
