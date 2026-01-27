#!/usr/bin/env bash

# Runs DockQ

set -euo pipefail

output_dir="$1"
reference_dir="$2"
shift; shift;
declare -a complexes=("$@")

if (( ${#complexes[@]} == 0 )); then
    echo "No complexes specified, processing all of them"
    complexes=($(find "${output_dir}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))
fi
echo "Processing ${#complexes[@]} complexes."

for complex_name in "${complexes[@]}"; do
    echo "Processing complex: ${complex_name}"


    exec &> >(tee "${output_dir}/${complex_name}/dockq.log")

    for model_pdb in $(find "${output_dir}/${complex_name}" -type f -name "*_model_*.pdb"); do
        reference_pdb="${reference_dir}/${complex_name}.pdb"

        # Run DockQ
        dockq_output_json="${model_pdb%.pdb}_dockq.json"
        echo "Running DockQ for model ${model_pdb} against reference ${reference_pdb}"
        DockQ --json "${dockq_output_json}" "${model_pdb}" "${reference_pdb}" > /dev/null

        echo "DockQ results for ${model_pdb} saved to ${dockq_output_json}"
    done
done
