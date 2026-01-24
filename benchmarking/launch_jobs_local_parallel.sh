#!/usr/bin/env bash

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
RUN_NAME="${RUN_NAME:-$(TZ=UTC date +%F-%H-%M-%S)}"

INPUT_DIR="${INPUT_DIR:-/data/alphafold-inputs/features}"
DATA_DIR="${DATA_DIR:-/data/alphafold}"
OUTPUT_DIR="${OUTPUT_DIR:-output/local/${RUN_NAME}}"

if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "Output directory ${OUTPUT_DIR} already exists." >&2
    exit 1
fi

echo "Making output directory ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

declare -a COMPLEXES=(
    "7fci"
    "7mnl"
    "7n0a"
    "7ox1"
    "7ox2"
    "7ox3"
    "7ox4"
    "7q6c"
    "7r58"
    "7ru6"
    "7sbd"
    "7sbg"
    "7sjo"
    "7sk3"
    "7sk4"
    "7sk5"
    "7sk6"
    "7sk7"
    "7sk8"
    "7sk9"
    "7so7"
    "7st8"
    "7t6x"
    "7t82"
    "7t9m"
    "7t9n"
    "7tuf"
    "7tug"
    "7u8c"
    "7u8g"
    "7uih"
    "7um3"
    "7ura"
    "7urc"
    "7urd"
    "7ure"
    "7uvf"
    "7vad"
    "7vae"
    "7vaf"
    "7vag"
    "7vgr"
    "7vgs"
    "7vn9"
    "7vng"
    "7w71"
    "7wsi"
    "7xq8"
    "7xy8"
    "7zlg"
    "7zlh"
    "7zli"
    "7zlj"
    "7zwi"
    "7zxf"
    "7zxg"
    "7zxk"
    "7zyi"
    "8cz5"
    "8dcy"
    "8ddk"
    "8djk"
    "8djm"
    "8dke"
    "8dki"
    "8dkm"
    "8dkw"
    "8dkx"
    "8hii"
    "8hij"
    "8hik"
)

declare -a SEEDS=($(seq 1 5))
declare -a MODEL_INDICES=($(seq 1 5))

# For testing
# COMPLEXES=("7u8c")
# SEEDS=(1)
# MODEL_INDICES=(1)

total_jobs="$(( ${#COMPLEXES[@]} * ${#SEEDS[@]} * ${#MODEL_INDICES[@]} ))"

echo "Launching ${total_jobs} jobs on 1 node:"
echo "- ${#COMPLEXES[@]} complexes x ${#SEEDS[@]} seeds x ${#MODEL_INDICES[@]} model indices"
echo "- ${TOTAL_GPUS} GPUs, ${TOTAL_CORES} cores (${CORES_PER_JOB} cores/job)"

JOBS_FILE="${OUTPUT_DIR}/jobs.txt"
# This shouldn't already exist, but clear it just in case
> "${JOBS_FILE}"

for seed in "${SEEDS[@]}"; do
    for model_index in "${MODEL_INDICES[@]}"; do
        for complex_name in "${COMPLEXES[@]}"; do
            printf "%s %s %s\n" "${complex_name}" "${seed}" "${model_index}" >> "${JOBS_FILE}"
        done
    done
done

# We put all the env vars in the explicit job invocation for reproducibility and
# so the full command is in the joblog file.

# exec so that any edits to the shell script don't confuse things
exec parallel --jobs "${TOTAL_GPUS}" \
    --colsep ' ' \
    --line-buffer \
    --joblog "${OUTPUT_DIR}/joblog.txt" \
    --bar \
    --halt now,fail=1 \
    "COMPLEX_NAME={1} SEED={2} MODEL_INDEX={3} SLOT={%} INPUT_DIR=${INPUT_DIR} DATA_DIR=${DATA_DIR} OUTPUT_DIR=${OUTPUT_DIR} CORES_PER_JOB=${CORES_PER_JOB} benchmarking/run_job.sh" \
    :::: "${JOBS_FILE}"
