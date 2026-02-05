#!/usr/bin/env bash

# Use gnu-parallel to launch a bunch of dockq jobs.

set -euo pipefail

if ! command -v parallel >/dev/null 2>&1; then
    echo "GNU parallel is required but not installed. Install it and retry." >&2
    exit 1
fi

RESULT_DIR="$1"
TOTAL_CORES="$(nproc --all)"

REFERENCE_DIR="${REFERENCE_DIR:-/data/alphafold-inputs/reference_pdbs}"
complexes=($(find "${RESULT_DIR}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))
# For testing
# complexes=("7u8c")

if ! [[ -d "${RESULT_DIR}" ]]; then
    echo "Output directory ${RESULT_DIR} doesn't exist." >&2
    exit 1
fi


total_jobs="${#complexes[@]}"
concurrent_jobs="$(( TOTAL_CORES / 2 ))"
if (( concurrent_jobs > total_jobs )); then
    concurrent_jobs="${total_jobs}"
fi

echo "Launching ${total_jobs} jobs on 1 node with ${concurrent_jobs} concurrent jobs."

JOBLOG_FILE="${RESULT_DIR}/dockq_joblog.txt"
> "${JOBLOG_FILE}"

set +e
time parallel \
    --jobs "${concurrent_jobs}" \
    --joblog "${JOBLOG_FILE}" \
    --bar \
    "benchmarking/dockq.sh ${RESULT_DIR} ${REFERENCE_DIR} {} &> /dev/null" \
    ::: "${complexes[@]}"

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
