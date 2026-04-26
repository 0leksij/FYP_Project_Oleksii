# Benchmarking Text-to-3D AI Models for Real-World Applications

**Final Year Project - Oleksii Shvets, UCD School of Computer Science**  
Supervisor: Professor Anh Vu Vo

---

## What this project does

Modern AI systems can generate 3D objects from a text description - you type "a wooden chair" and the model produces a 3D mesh. This project asks: **how good are those meshes actually?**

Not just "do they look okay in a render" - but are they usable in the real world? Can you 3D-print them? Import them into CAD tools? Use them in a game engine? Run physics simulations on them?

This repository contains a complete, reproducible benchmark pipeline that evaluates text-to-3D models across four dimensions:

- **Geometric complexity** - how detailed is the mesh?
- **Topological integrity** - is the mesh structurally sound (no holes, no floating fragments)?
- **Semantic alignment** - does it actually look like what you asked for?
- **Human preference** - would a person find it visually acceptable?

Models evaluated: **Point-E**, **Shap-E**, **Stable-DreamFusion**, and additional community models, compared against a hand-crafted reference mesh.

---

## Repository structure

```
FYP_Project_Oleksii/
├── Scripts/                    # Generation scripts (run these first)
│   ├── generate_point_e.py
│   ├── generate_shap_e.py
│   └── generate_dreamfusion.sh
├── Assets/                     # Generated meshes go here (organised by model)
│   ├── point_e/
│   ├── shap_e/
│   ├── dreamfusion/
│   └── nonAI/
├── benchmark/
│   ├── evaluate.py             # Core evaluation pipeline
│   ├── metrics/                # Individual metric modules
│   │   ├── mesh_metrics.py
│   │   ├── clip_metrics.py
│   │   └── image_reward_metrics.py
│   ├── scripts/
│   │   └── evaluate_all.py     # Batch evaluation driver
│   ├── visualise_results.ipynb # Charts and radar plots
│   ├── results/                # CSV, JSON, and PNG outputs
│   └── requirements.txt
├── Models/                     # Git submodules (Point-E, Shap-E, CLIP, DreamFusion)
└── report.tex                  # Full project report (LaTeX)
```

---

## Quick start

### 1. Set up the environment

```bash
conda create -n t2d-benchmark python=3.10
conda activate t2d-benchmark
pip install -r benchmark/requirements.txt
```

Headless rendering (GPU server without a display) requires EGL:

```bash
export PYOPENGL_PLATFORM=egl
```

### 2. Generate meshes

Run one or more generation scripts. Each script saves a `.ply` file into the corresponding `Assets/` subdirectory.

**Point-E** (fastest, ~30 seconds on GPU):
```bash
python Scripts/generate_point_e.py --prompt "a simple wooden chair"
```

**Shap-E** (~1-2 minutes on GPU):
```bash
python Scripts/generate_shap_e.py --prompt "a simple wooden chair"
```

**Stable-DreamFusion** (slow, ~20-40 minutes on GPU):
```bash
bash Scripts/generate_dreamfusion.sh "a simple wooden chair"
```

All scripts accept `--prompt` and `--output` arguments. Run `--help` for details.

### 3. Run the benchmark

```bash
cd benchmark
python scripts/evaluate_all.py --prompt "a simple wooden chair" --output-name wooden_chair
```

This scans every subdirectory in `Assets/`, evaluates each mesh it finds, and writes results to `benchmark/results/wooden_chair.csv` and `benchmark/results/wooden_chair.json`.

To skip rendering (geometry metrics only, much faster):
```bash
python scripts/evaluate_all.py --prompt "a simple wooden chair" --skip-rendering
```

### 4. Visualise results

Open the notebook and run all cells:
```bash
cd benchmark
jupyter notebook visualise_results.ipynb
```

The notebook reads `results/*.csv` and produces bar charts, error-bar plots, and per-model radar charts.

---

## Adding your own model

1. Generate a mesh in any format supported by Trimesh (`.ply`, `.obj`, `.glb`).
2. Create a new subdirectory under `Assets/` named after your model:
   ```
   Assets/my_model/a_simple_wooden_chair.ply
   ```
3. Re-run `evaluate_all.py` with the same prompt - it will pick up the new model automatically.

---

## Understanding the metrics

The benchmark reports three categories of metrics. Here is what each one means in plain language and why it matters.

### Geometry metrics

| Metric | What it measures | What a good value looks like |
|---|---|---|
| `num_vertices` / `num_faces` | How many polygons the mesh has | Depends on use case; 10k-100k is typical for AI models |
| `surface_area` | Total surface area in model-space units | Comparable values across models for the same object |
| `volume` | Interior volume (only valid for closed meshes) | Non-zero if the mesh is watertight |
| `bbox_x/y/z` | Bounding box dimensions | Proportions should match the real object (e.g. a chair is taller than it is wide) |

**Tip:** If `volume` is 0 or very small, the mesh is not closed and volume cannot be computed reliably.

---

### Topology metrics - the most important for real-world use

These tell you whether the mesh is actually usable, not just whether it looks good.

| Metric | What it measures | What a good value looks like |
|---|---|---|
| `is_watertight` | Does the mesh form a completely closed surface with no holes? | `True` - required for 3D printing, FEM simulation, CAD Boolean operations |
| `num_components` | How many disconnected pieces the mesh has | `1` - a single connected body. Values above 1 mean the object is fragmented |
| `num_nonmanifold_edges` | Edges shared by more than two faces (illegal geometry) | `0` - any non-zero value will break UV unwrapping, Boolean operations, and remeshing |
| `normal_consistency` | What fraction of adjacent face pairs have normals pointing the same way | Close to `1.0` - values below 0.9 cause dark patches and rendering artefacts |

**What these failures mean in practice:**

- **Not watertight** → cannot 3D-print, cannot run physics simulation, cannot do CAD Boolean operations (union/subtract)
- **Multiple components** → each disconnected piece must be handled separately; many tools reject fragmented meshes entirely
- **Non-manifold edges** → Boolean operations crash or produce garbage; UV unwrapping fails
- **Low normal consistency** → half the mesh may appear invisible or dark depending on the viewing angle

---

### Mesh regularity metrics

These measure the *quality* of the tessellation - how evenly the surface is subdivided - independently of whether the shape is correct.

| Metric | What it measures | What a good value looks like |
|---|---|---|
| `face_aspect_ratio_mean` | Average ratio of longest to shortest edge per triangle. 1.0 = perfect equilateral triangle | 1.5-3.0 is typical; above 10 causes FEM solver errors |
| `face_aspect_ratio_max` | Worst single triangle in the mesh | Below 10; very high values indicate at least one degenerate "needle" triangle |
| `face_area_cv` | How uneven the face sizes are (standard deviation / mean) | Low values = uniform mesh. AI meshes often score 100-10000+ (highly non-uniform) |
| `edge_length_cv` | Same idea applied to edge lengths | Low = uniform; high = some edges are micro-sized, others are huge |

**Why this matters:** Simulation tools (FEM, fluid dynamics, physics engines) require roughly uniform face sizes to produce numerically accurate results. A mesh with CV of 1000+ cannot be used directly in simulation without re-meshing.

---

### Semantic and perceptual metrics

These measure whether the object visually matches the text prompt.

| Metric | What it measures | What a good value looks like |
|---|---|---|
| `clip_score_mean` | Cosine similarity between renders and the text prompt, using the CLIP ViT-B/32 model | Typically 0.2-0.35 for 3D renders; higher = better match |
| `clip_score_std` | Variation in CLIP score across views | Low = consistent from all angles; high = object looks different from different sides |
| `multiview_consistency` | Mean pairwise CLIP similarity between all rendered views | Above 0.9 for a coherent object; low values suggest a "Janus problem" (multiple fronts) |
| `image_reward_mean` | Human preference score from the ImageReward model (trained on 137k human comparisons) | Negative values are normal for 3D renders; higher (less negative) = more visually acceptable |
| `image_reward_std` | Variation in preference score across views | Low = consistently good/bad from all angles |

**Interpreting CLIP scores:** All models for the same object class tend to cluster within a narrow 0.01-0.02 range. CLIP cannot reliably distinguish a clean mesh from a noisy one - it only checks rough semantic category. Use ImageReward to distinguish quality within the same category.

**Interpreting ImageReward:** Scores are calibrated against photorealistic 2D images, so 3D renders always score lower than real photos. A score of -0.3 is excellent for a generated 3D mesh; -2.0 is poor. The relative ranking between models is reliable even though the absolute values are negative.

**The Janus problem:** A model that looks like a chair from the front but a blob from the side will have high `clip_score_mean` (each view triggers "chair" in CLIP) but low `multiview_consistency` (different views look very different to each other). This is the main artefact of SDS-based methods like DreamFusion.

---

### Reference-based metric

| Metric | What it measures |
|---|---|
| `chamfer_distance` | Average distance between points on the generated mesh and points on a reference mesh - lower is better |

This metric only runs when a reference mesh exists. It measures geometric similarity to a known-correct shape, independently of polygon count or topology.

---

## Model weights (not included in the repo)

Model weight files (`.pt`, `.pth`) are excluded from git because they are several gigabytes each. They are downloaded automatically on first run.

**If you need to pre-download them manually:**

**Point-E weights** (~900 MB for base model + SDF model):
```bash
conda activate t2d-benchmark
python -c "
import sys; sys.path.insert(0, 'Models/point-e')
from point_e.models.download import load_checkpoint
load_checkpoint('base40M-textvec', cache_dir='Scripts/point_e_model_cache')
load_checkpoint('sdf', cache_dir='Scripts/point_e_model_cache')
"
```

**Shap-E weights** (~3 GB for transmitter + text model):
```bash
conda activate t2d-benchmark
python -c "
import sys; sys.path.insert(0, 'Models/shap-e')
from shap_e.models.download import load_model
load_model('transmitter', device='cpu')
load_model('text300M', device='cpu')
"
```

**CLIP ViT-B/32** (downloaded automatically by the `clip` package on first use, ~350 MB):
```bash
conda activate t2d-benchmark
python -c "import clip; clip.load('ViT-B/32')"
```

**Stable-DreamFusion** uses Stable Diffusion weights via `diffusers` - downloaded automatically on first training run. Requires a Hugging Face account and `huggingface-cli login`.

All weights are cached locally and reused on subsequent runs.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU works for evaluation only; generation requires GPU)
- ~10 GB free disk space for all model weights
- For headless rendering on a server: `PYOPENGL_PLATFORM=egl` and EGL libraries installed (`apt install libegl1`)

---
