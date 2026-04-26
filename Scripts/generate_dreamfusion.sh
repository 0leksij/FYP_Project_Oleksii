#!/bin/bash
# Generate a 3D mesh using stable-dreamfusion.
#
# Usage:
#   bash generate_dreamfusion.sh
#   bash generate_dreamfusion.sh "a red fire hydrant" "../Assets/dreamfusion/fire_hydrant.ply"
set -e

PROMPT="${1:-a simple wooden chair}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$REPO_ROOT/Models/stable-dreamfusion"

# Derive a slug for workspace and output path
SLUG="${PROMPT// /_}"
SLUG="${SLUG:0:40}"

OUTPUT_PLY="${2:-$REPO_ROOT/Assets/dreamfusion/${SLUG}.ply}"
WORKSPACE="$(dirname "$OUTPUT_PLY")/workspace_${SLUG}"

mkdir -p "$(dirname "$OUTPUT_PLY")"

echo "Prompt:  $PROMPT"
echo "Output:  $OUTPUT_PLY"
echo "Running DreamFusion training..."

python "$MODEL_DIR/main.py" \
    --text "$PROMPT" \
    --workspace "$WORKSPACE" \
    --iters 2000 \
    --resolution 64 \
    --cuda_ray

echo "Extracting mesh..."
python "$MODEL_DIR/scripts/extract_mesh.py" \
    --ckpt "$WORKSPACE/checkpoints/latest.pth" \
    --mesh_path "$OUTPUT_PLY"

echo "Saved: $OUTPUT_PLY"