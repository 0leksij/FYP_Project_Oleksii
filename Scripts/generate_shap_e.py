#!/usr/bin/env python3
"""Generate a 3D mesh from text using Shap-E.

Usage:
    python generate_shap_e.py
    python generate_shap_e.py --prompt "a red fire hydrant" --output ../Assets/shap_e/fire_hydrant.ply
"""

import argparse
import sys
import torch
import trimesh
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

import os
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "Scripts"))

sys.path.insert(0, str(REPO_ROOT / "Models" / "shap-e"))

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a 3D mesh with Shap-E")
    parser.add_argument(
        "--prompt", "-p",
        default="a simple wooden chair",
        help="Text prompt to generate from (default: 'a simple wooden chair')",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output .ply path (default: Assets/shap_e/<prompt_slug>.ply)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=15.0,
        help="Classifier-free guidance scale (default: 15.0)",
    )
    parser.add_argument(
        "--steps", type=int, default=64,
        help="Number of Karras diffusion steps (default: 64)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    prompt = args.prompt
    device = args.device

    if args.output is None:
        slug = prompt.replace(" ", "_")[:40]
        output_path = REPO_ROOT / "Assets" / "shap_e" / f"{slug}.ply"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device:  {device}")
    print(f"Prompt:  {prompt}")
    print(f"Output:  {output_path}")
    print("Loading Shap-E models...")

    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    print("Generating latent...")
    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=args.guidance_scale,
        model_kwargs=dict(texts=[prompt]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=args.steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    print("Decoding mesh...")
    tri_mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
    mesh = trimesh.Trimesh(vertices=tri_mesh.verts, faces=tri_mesh.faces)
    mesh.export(str(output_path))

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()