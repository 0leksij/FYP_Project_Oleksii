#!/usr/bin/env python3
"""
Batch evaluation driver — evaluates all generated meshes in Assets/.

Usage (run from the benchmark/ directory):
    python scripts/evaluate_all.py --prompt "a simple wooden chair"
    python scripts/evaluate_all.py --prompt "a red fire hydrant" --output-name fire_hydrant

The script scans Assets/ for .ply files organised by model name:
    Assets/
        point_e/    <prompt_slug>.ply
        shap_e/     <prompt_slug>.ply
        dreamfusion/<prompt_slug>.ply
        ...

Each subfolder is treated as one model; the mesh_id is the subfolder name.
All meshes in a single run are assumed to have been generated from the same prompt.
"""

import sys
import argparse
from pathlib import Path

# Allow imports from the parent benchmark/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import MeshEvaluationPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "Assets"


def collect_meshes(assets_dir: Path, prompt: str):
    """
    Scan assets_dir for .ply files.  Each immediate subdirectory is one model.
    Returns a list of dicts ready for MeshEvaluationPipeline.evaluate_batch().
    """
    mesh_list = []
    for model_dir in sorted(assets_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        ply_files = sorted(model_dir.glob("*.ply"))
        if not ply_files:
            continue
        # Use the most recently modified .ply if there are several
        mesh_file = max(ply_files, key=lambda p: p.stat().st_mtime)
        mesh_list.append({
            'mesh_id':   model_dir.name,
            'mesh_path': str(mesh_file),
            'prompt':    prompt,
        })
    return mesh_list


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all generated meshes in Assets/")
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="The text prompt that was used to generate the meshes",
    )
    parser.add_argument(
        "--assets-dir",
        default=str(ASSETS_DIR),
        help=f"Directory containing generated meshes (default: {ASSETS_DIR})",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base name for output CSV/JSON files (default: derived from prompt)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "results"),
        help="Directory to save evaluation results (default: benchmark/results/)",
    )
    parser.add_argument(
        "--render-dir",
        default=str(Path(__file__).resolve().parent.parent / "results" / "renders"),
        help="Directory to save rendered views (default: benchmark/results/renders/)",
    )
    parser.add_argument(
        "--skip-rendering", action="store_true",
        help="Skip rendering and CLIP/ImageReward scoring (mesh metrics only)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_name = args.output_name or args.prompt.replace(" ", "_")[:40]
    assets_dir  = Path(args.assets_dir)

    print("=" * 60)
    print("Text-to-3D Mesh Evaluation Pipeline")
    print("=" * 60)
    print(f"Prompt:      {args.prompt}")
    print(f"Assets dir:  {assets_dir}")
    print(f"Output dir:  {args.output_dir}")

    mesh_list = collect_meshes(assets_dir, args.prompt)
    if not mesh_list:
        print(f"\nNo .ply files found in {assets_dir}")
        print("Run a generation script first, e.g.:")
        print("  python Scripts/generate_point_e.py --prompt \"...\"")
        sys.exit(1)

    print(f"\nFound {len(mesh_list)} mesh(es):")
    for m in mesh_list:
        print(f"  {m['mesh_id']:20s}  {m['mesh_path']}")

    pipeline = MeshEvaluationPipeline(
        output_dir=args.output_dir,
        render_dir=args.render_dir,
        verbose=True,
    )

    print("\nStarting evaluation...")
    results_df = pipeline.evaluate_batch(
        mesh_list,
        output_name=output_name,
        skip_rendering=args.skip_rendering,
    )

    print("\nGenerating summary report...")
    pipeline.generate_summary_report(results_df, output_name=f"{output_name}_summary")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results: {args.output_dir}/{output_name}.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()