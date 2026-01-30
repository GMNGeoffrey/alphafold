#!/usr/bin/env bash

# Launch a benchmarking job for a single complex-seed-model combination. This is
# constructed to be launched with the accompanying gnu-parallel
# launch_benchmark_jobs_local_parallel.sh script, but can be run directly by
# setting the correct environment variables. It prioritizes a consistent
# benchmarking environment over maximum performance (e.g. by using taskset).

set -euo pipefail

# Required environment variables:
# Variable per job:
#   COMPLEX_NAME: Name of the protein complex to benchmark (e.g., "7fci").
#   MODEL_INDEX: Index of the Alphafold model to run (1-5 for multimer_v3).
#   STARTING_SEED: Starting random seed to use for this job.
#   SEED_COUNT: Number of seeds to run for this complex.
#   SLOT: The parallel slot number (1 to TOTAL_GPUS) assigned by GNU parallel.
# The same for all jobs in a run
#   INPUT_DIR: Directory containing input feature .npz files.
#   DATA_DIR: Directory containing Alphafold model parameter files under params/ subdir.
#   OUTPUT_DIR: Root directory to write output results to.
#   CORES_PER_JOB: Number of CPU cores allocated per job (should be nproc // TOTAL_GPUS).
#   FIXED_RECYCLES: If set to a non-empty string, use this fixed number of
#       recycles instead of the default limit with early stopping behavior.
#       To avoid mistakes, this must be set either way. Use an explicit empty
#       string to get the default behavior.

GPU="$((SLOT - 1))"
CPU_START="$(((SLOT - 1) * CORES_PER_JOB))"
CPU_END="$((CPU_START + CORES_PER_JOB - 1))"

MODEL_NAME="model_${MODEL_INDEX}_multimer_v3"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${COMPLEX_NAME}/seed_${STARTING_SEED}/model_${MODEL_INDEX}"

mkdir -p "${RUN_OUTPUT_DIR}"


declare -a ARGS=(
  --data_dir="${DATA_DIR}"
  --feature_paths="${INPUT_DIR}/${COMPLEX_NAME}.npz"
  --output_dir="${RUN_OUTPUT_DIR}"
  --model_preset=multimer
  --model_names="${MODEL_NAME}"
  --random_seed="${STARTING_SEED}"
  --consistent_random_seeds
  --num_multimer_predictions_per_model="${SEED_COUNT}"
  --models_to_relax=none
  --use_gpu_relax
  --save_full_results=False
)

if [[ -n "${FIXED_RECYCLES}" ]]; then
  ARGS+=(
    --num_recycle="${FIXED_RECYCLES}"
    --recycle_early_stop_tolerance=-1
  )
fi

(
  set -x;
  CUDA_VISIBLE_DEVICES="${GPU}" taskset -c "${CPU_START}-${CPU_END}" \
      python run_alphafold_model_only.py "${ARGS[@]}"
) 2>&1 | tee "${RUN_OUTPUT_DIR}/output.log"
