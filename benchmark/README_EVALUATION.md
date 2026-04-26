# Mesh Evaluation Pipeline

## Installation
```bash
conda activate t2d-benchmark
pip install -r requirements.txt
```

## Quick Start

### Evaluate a single mesh:
```python
from src.evaluate import MeshEvaluationPipeline

pipeline = MeshEvaluationPipeline()
result = pipeline.evaluate_single_mesh(
    mesh_path="data/generated/chair.ply",
    prompt="a wooden chair",
    mesh_id="chair_001"
)
print(result)
```

### Evaluate all meshes:
```bash
python scripts/evaluate_all.py
```

## Output Files

- `results/evaluation/all_meshes.csv` - All metrics in CSV format
- `results/evaluation/all_meshes.json` - All metrics in JSON format
- `results/evaluation/summary.json` - Summary statistics
- `results/renders/` - Rendered multi-view images

## Metrics Computed

**Geometry:**
- num_vertices, num_faces, surface_area, volume
- bbox_x, bbox_y, bbox_z, bbox_volume

**Topology:**
- is_watertight, num_components
- num_nonmanifold_edges, normal_consistency

**Semantic:**
- clip_score_mean, clip_score_std, clip_score_min, clip_score_max

**Comparison (if reference provided):**
- chamfer_distance