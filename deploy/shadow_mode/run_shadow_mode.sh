#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/pipeline_config.yaml}
OUT_DIR=${2:-output/shadow}

mkdir -p "$OUT_DIR"
echo "Running pipeline in shadow mode with config=$CONFIG -> out=$OUT_DIR"

# Example invocation of existing pipeline runner; adapt flags as needed
python run_pipeline.py --config "$CONFIG" --out "$OUT_DIR" --shadow

echo "Shadow run complete. Scores and alerts stored in $OUT_DIR"
