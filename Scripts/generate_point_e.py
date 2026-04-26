#!/usr/bin/env python3
"""Generate a 3D mesh from text using Point-E.

Usage:
    python generate_point_e.py
    python generate_point_e.py --prompt "a red fire hydrant" --output ../Assets/point_e/fire_hydrant.ply
"""

import argparse
import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Use local cache dirs to avoid repeat downloads
import os
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "Scripts"))

sys.path.insert(0, str(REPO_ROOT / "Models" / "point-e"))

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh


CACHE_DIR = REPO_ROOT / "Models" / "point-e" / "point_e" / "examples" / "point_e_model_cache"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a 3D mesh with Point-E")
    parser.add_argument(
        "--prompt", "-p",
        default="a simple wooden chair",
        help="Text prompt to generate from (default: 'a simple wooden chair')",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output .ply path (default: Assets/point_e/<prompt_slug>.ply)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=128,
        help="Marching-cubes grid resolution (default: 128)",
    )
    return parser.parse_args()


def load_from_cache(name: str, filename: str, device: str):
    path = CACHE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Cached model not found: {path}\n"
            f"Download with: python -c \"from point_e.models.download import load_checkpoint; "
            f"load_checkpoint('{name}')\""
        )
    return torch.load(path, map_location=device)


def main():
    args = parse_args()

    prompt = args.prompt
    device = args.device

    if args.output is None:
        slug = prompt.replace(" ", "_")[:40]
        output_path = REPO_ROOT / "Assets" / "point_e" / f"{slug}.ply"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device:  {device}")
    print(f"Prompt:  {prompt}")
    print(f"Output:  {output_path}")
    print("Loading Point-E models...")

    base_name = "base40M-textvec"
    base_model = model_from_config(MODEL_CONFIGS[base_name], device=device)
    base_model.eval()
    base_model.load_state_dict(load_from_cache(base_name, "base_40m_textvec.pt", device))

    sdf_model = model_from_config(MODEL_CONFIGS["sdf"], device=device)
    sdf_model.eval()
    sdf_model.load_state_dict(load_from_cache("sdf", "sdf.pt", device))

    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    sampler = PointCloudSampler(
        device=device,
        models=[base_model],
        diffusions=[base_diffusion],
        num_points=[1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0],
        use_karras=[True],
        karras_steps=[64],
        sigma_min=[1e-3],
        sigma_max=[160],
        s_churn=[0],
    )

    print(f"Generating point cloud...")
    samples = None
    for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]

    print("Converting point cloud to mesh...")
    mesh = marching_cubes_mesh(
        pc=pc,
        model=sdf_model,
        batch_size=4096,
        grid_size=args.grid_size,
        progress=True,
    )

    with open(output_path, "wb") as f:
        mesh.write_ply(f)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()