#!/usr/bin/env python3
"""Test the evaluation pipeline with a simple mesh"""

import sys
from pathlib import Path
import trimesh
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import MeshEvaluationPipeline

# Create a simple test mesh (cube)
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
])

faces = np.array([
    [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Save test mesh
test_dir = Path("data/test")
test_dir.mkdir(parents=True, exist_ok=True)
mesh.export(test_dir / "test_cube.ply")

print("Created test mesh")

# Run evaluation
pipeline = MeshEvaluationPipeline(
    output_dir="results/test",
    render_dir="results/test_renders"
)

result = pipeline.evaluate_single_mesh(
    mesh_path=str(test_dir / "test_cube.ply"),
    prompt="a cube",
    mesh_id="test_cube"
)

print("\n=== Test Results ===")
for key, value in result.items():
    print(f"{key}: {value}")

print("\nTest complete!")