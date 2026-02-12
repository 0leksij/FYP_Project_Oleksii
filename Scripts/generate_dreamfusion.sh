#!/bin/bash
#remember to chmod +x generate_dreamfusion.sh

PROMPT="a small concrete bus stop shelter"
OUTPUT_DIR="../Assets/dreamfusion"

mkdir -p $OUTPUT_DIR

cd ../Models/stable-dreamfusion

python main.py \
  --text "$PROMPT" \
  --workspace ../../Assets/dreamfusion/workspace \
  --iters 2000 \
  --resolution 64 \
  --cuda_ray

python scripts/extract_mesh.py \
  --ckpt ../../Assets/dreamfusion/workspace/checkpoints/latest.pth \
  --mesh_path ../../Assets/dreamfusion/output.ply

echo "DreamFusion mesh saved to ../../Assets/dreamfusion/output.ply"
