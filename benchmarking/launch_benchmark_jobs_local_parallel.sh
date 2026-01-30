#!/usr/bin/env bash

# Use gnu-parallel to launch a bunch of alphafold jobs. This probably isn't the
# best approach for an actual production use case, where you'd either want to
# use kubernetes or probably a python-based system with more shared resources
# between jobs, but for benchmarking we prioritize a consistent environment
# where some efficiencies of sharing are not desirable because they introduce
# variability (e.g. we use taskset within each job).

set -euo pipefail

if ! command -v parallel >/dev/null 2>&1; then
    echo "GNU parallel is required but not installed. Install it and retry." >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null; then
    TOTAL_GPUS="$(nvidia-smi --list-gpus | wc -l)"
elif command -v amd-smi >/dev/null; then
    TOTAL_GPUS="$(amd-smi list | grep -c "^GPU" || true)"
else
    echo "Neither nvidia-smi nor amd-smi found. Cannot detect GPUs." >&2
    exit 1
fi

if ! (( TOTAL_GPUS > 0 )); then
    echo "No GPUs found (TOTAL_GPUS=${TOTAL_GPUS})" >&2
    exit 1
fi
TOTAL_CORES="$(nproc --all)"

if (( TOTAL_CORES % TOTAL_GPUS != 0 )); then
    echo "Total cores (${TOTAL_CORES}) is not divisible by ${TOTAL_GPUS}." >&2
    echo "Each job must receive exactly 1/${TOTAL_GPUS} of the cores." >&2
    exit 1
fi

CORES_PER_JOB="$(( TOTAL_CORES / TOTAL_GPUS ))"

INPUT_DIR="${INPUT_DIR:-/data/alphafold-inputs/features}"
DATA_DIR="${DATA_DIR:-/data/alphafold}"
OUTPUT_DIR="${OUTPUT_DIR:-output/local/$(TZ=UTC date +%F-%H-%M-%S)}"

if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "Output directory ${OUTPUT_DIR} already exists." >&2
    exit 1
fi

echo "Making output directory ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# We sort complexes largest first to try to get a more even distribution of work
# (LPT greedy scheduling, effectively)
readarray -t COMPLEXES < <(
    find "${INPUT_DIR}" -maxdepth 1 -name '*.npz' -type f -printf '%s %f\n' \
        | sort -rn \
        | cut -d ' ' -f 2 \
        | xargs basename --multiple --suffix=.npz \
)


declare -a MODEL_INDICES=($(seq 1 5))
STARTING_SEED=1
SEED_COUNT=5

# For testing
# COMPLEXES=("7fci" "7u8c")
# SEED_COUNT=1
# MODEL_INDICES=(1)

JOB_COUNT="$(( ${#COMPLEXES[@]} * ${#MODEL_INDICES[@]} ))"

TOTAL_PREDICTIONS="$(( ${#COMPLEXES[@]} * SEED_COUNT * ${#MODEL_INDICES[@]} ))"
echo "Launching ${TOTAL_PREDICTIONS} predictions as ${JOB_COUNT} jobs on 1 node:"
echo "- ${#COMPLEXES[@]} complexes x ${SEED_COUNT} seeds x ${#MODEL_INDICES[@]} model indices"
echo "- ${TOTAL_GPUS} GPUs, ${TOTAL_CORES} cores (${CORES_PER_JOB} cores/job)"

# We put all the env vars in the explicit job invocation so we get the full
# command in the joblog file for reproducibility.

# We want to print something if there are errors, so disable errexit
set +e
parallel \
    --jobs "${TOTAL_GPUS}" \
    --colsep ' ' \
    --joblog "${OUTPUT_DIR}/joblog.txt" \
    --bar \
    "SLOT={%} COMPLEX_NAME={1} MODEL_INDEX={2} FIXED_RECYCLES=${FIXED_RECYCLES} STARTING_SEED=${STARTING_SEED} SEED_COUNT=${SEED_COUNT} INPUT_DIR=${INPUT_DIR} DATA_DIR=${DATA_DIR} OUTPUT_DIR=${OUTPUT_DIR} CORES_PER_JOB=${CORES_PER_JOB} benchmarking/run_benchmark_job.sh &> /dev/null" \
    ::: "${COMPLEXES[@]}" ::: "${MODEL_INDICES[@]}"

ret=$?
if (( ret != 0 )); then
    if (( ret <= 100 )); then
        echo "${ret} jobs failed" >&2
    elif (( ret == 101 )); then
        echo "More than 100 jobs failed" >&2
    else
        echo "GNU parallel failed with error code ${ret}" >&2
    fi
    exit "${ret}"
else
    echo "All jobs completed successfully."
fi
