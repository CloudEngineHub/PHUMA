#!/bin/bash
set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$SCRIPT_DIR"

echo "Project directory: $project_dir"

# Download PHUMA(without LAFAN1 and LocoMuJoCo) dataset
hf download DAVIAN-Robotics/PHUMA --repo-type dataset --local-dir "$project_dir"
unzip "$project_dir/data.zip" -d "$project_dir"
rm "$project_dir/data.zip"

# Download LAFAN1 and LocoMuJoCo datasets
PYTHONPATH="$project_dir/src:$PYTHONPATH" python "$project_dir/src/utils/download_lafan_locomujoco.py" --project_dir "$project_dir" --remove_original

# Preprocess G1 folder
PYTHONPATH="$project_dir/src:$PYTHONPATH" python "$project_dir/src/curation/preprocess_g1_folder.py" --project_dir "$project_dir"

# Preprocess H1_2 folder
PYTHONPATH="$project_dir/src:$PYTHONPATH" python "$project_dir/src/curation/preprocess_h1_2_folder.py" --project_dir "$project_dir"

# Remove preprocessed original datasets
rm -rf "$project_dir/data/preprocessed_original"