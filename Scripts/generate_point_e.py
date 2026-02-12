import os
import torch
from point_e.diffusion.configs import DIFFUSION_CONFIGS, model_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
from point_e.util.mesh import write_ply

# ===== USER CONFIG =====
PROMPT = "a small concrete bus stop shelter"
OUTPUT_ROOT = "../Assets/point_e"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =======================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("Loading Point-E models...")

base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device=DEVICE)
base_model.load_state_dict(torch.load(f"../Models/point-e/point_e/models/{base_name}.pt"))

diffusion = DIFFUSION_CONFIGS[base_name]
sampler = PointCloudSampler(
    device=DEVICE,
    models=[base_model],
    diffusions=[diffusion],
    num_points=[1024],
    aux_channels=['R', 'G', 'B'],
)

print("Generating point cloud...")
samples = None
for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[PROMPT])):
    samples = x

pc = sampler.output_to_point_clouds(samples)[0]

print("Converting to mesh...")
mesh = marching_cubes_mesh(pc)

output_path = os.path.join(OUTPUT_ROOT, "output.ply")
write_ply(mesh, output_path)

print(f"Saved to {output_path}")
