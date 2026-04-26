#!/usr/bin/env python3
"""
CLIP-based semantic alignment metrics
"""

import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Dict, Optional
import trimesh
import warnings

class CLIPEvaluator:
    """Evaluate semantic alignment using CLIP"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize CLIP model
        
        Args:
            device: 'cuda' or 'cpu', auto-detected if None
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
    def compute_text_image_similarity(self, 
                                      image_path: str, 
                                      text_prompt: str) -> float:
        """
        Compute CLIP similarity between image and text
        
        Args:
            image_path: Path to rendered image
            text_prompt: Text description
            
        Returns:
            Similarity score (0-1, higher is better)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text_input = clip.tokenize([text_prompt]).to(self.device)
            
            # Compute embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_features @ text_features.T).item()
                
            return float(similarity)
            
        except Exception as e:
            warnings.warn(f"CLIP evaluation failed: {e}")
            return 0.0
    
    def compute_multiview_clip_score(self,
                                     image_paths: List[str],
                                     text_prompt: str) -> Dict[str, float]:
        """
        Compute CLIP scores across multiple views
        
        Args:
            image_paths: List of paths to rendered views
            text_prompt: Text description
            
        Returns:
            Dictionary with mean, std, min, max CLIP scores
        """
        scores = []
        
        for img_path in image_paths:
            score = self.compute_text_image_similarity(img_path, text_prompt)
            scores.append(score)
        
        if len(scores) == 0:
            return {
                'clip_score_mean': 0.0,
                'clip_score_std': 0.0,
                'clip_score_min': 0.0,
                'clip_score_max': 0.0
            }
        
        return {
            'clip_score_mean': float(np.mean(scores)),
            'clip_score_std': float(np.std(scores)),
            'clip_score_min': float(np.min(scores)),
            'clip_score_max': float(np.max(scores))
        }

    def compute_multiview_consistency(self, image_paths: List[str]) -> float:
        """
        Measure 3D coherence by computing mean pairwise CLIP similarity
        across all rendered views.

        High score (~1.0) = all views look like the same object (good 3D consistency).
        Low score         = views look unrelated (e.g. Janus problem, noisy geometry).
        """
        if len(image_paths) < 2:
            return 0.0

        embeddings = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model.encode_image(image_input)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat.squeeze(0))
            except Exception:
                continue

        if len(embeddings) < 2:
            return 0.0

        embs = torch.stack(embeddings)          # (n, d)
        sim_matrix = (embs @ embs.T).cpu()      # (n, n)
        n = len(embeddings)
        # mean of upper triangle, excluding diagonal
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        return float(sim_matrix[mask].mean().item())


def render_multiview_images(mesh_path: str,
                           output_dir: str,
                           num_views: int = 8,
                           resolution: int = 512) -> List[str]:
    """
    Render mesh from multiple viewpoints using pyrender with EGL (headless GPU).

    Args:
        mesh_path: Path to mesh file
        output_dir: Directory to save rendered images
        num_views: Number of views to render
        resolution: Image resolution

    Returns:
        List of paths to rendered images
    """
    import os
    # Must be set before any OpenGL import
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.makedirs(output_dir, exist_ok=True)

    try:
        import pyrender

        # Load, centre and normalise mesh
        mesh = trimesh.load(mesh_path, force='mesh', process=True)
        # Fix inconsistent normals (common in generated meshes)
        mesh.fix_normals()
        # Use bounding box center (not area-weighted centroid) for visual centering
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.vertices -= center
        scale = max(mesh.extents)
        if scale > 0:
            mesh.vertices /= scale

        # doubleSided=True prevents backface culling, which causes ~half of
        # orbit views to appear blank when mesh normals are inconsistent
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.7, 0.7, 0.7, 1.0],
            doubleSided=True,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0])
        scene.add(pr_mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        renderer = pyrender.OffscreenRenderer(resolution, resolution)

        def make_pose(eye, target=np.zeros(3), world_up=np.array([0.0, 0.0, 1.0])):
            """Right-handed look-at pose (camera-to-world)."""
            forward = target - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, world_up)       # cross(forward, up) — NOT (up, forward)
            r_norm = np.linalg.norm(right)
            if r_norm < 1e-6:
                right = np.array([1.0, 0.0, 0.0])
            else:
                right /= r_norm
            cam_up = np.cross(right, forward)          # cross(right, forward) — NOT (forward, right)
            p = np.eye(4)
            p[:3, 0] = right
            p[:3, 1] = cam_up
            p[:3, 2] = -forward   # camera looks along its local -Z
            p[:3, 3] = eye
            return p

        rendered_paths = []
        distance = 2.5
        # Distribute views across two elevation rings for better coverage
        elevations = [0.6, 1.2]  # slight and medium elevation
        view_poses = []
        for elev in elevations:
            for az in np.linspace(0, 2 * np.pi, max(num_views // len(elevations), 4), endpoint=False):
                cx = distance * np.cos(az)
                cy = distance * np.sin(az)
                view_poses.append(np.array([cx, cy, elev]))
        # Trim to requested number of views
        view_poses = view_poses[:num_views]

        bg = np.array([255, 255, 255], dtype=np.uint8)  # white background colour

        for i, pos in enumerate(view_poses):
            pose = make_pose(pos)

            cam_node = scene.add(camera, pose=pose)
            light_node = scene.add(light, pose=pose)

            color, _ = renderer.render(scene)

            # Skip blank (all-background) renders — mesh not visible from this angle
            non_bg = (~np.all(color == bg, axis=-1)).sum()
            if non_bg == 0:
                scene.remove_node(cam_node)
                scene.remove_node(light_node)
                continue

            output_path = os.path.join(output_dir, f"view_{i:02d}.png")
            Image.fromarray(color).save(output_path)
            rendered_paths.append(output_path)

            scene.remove_node(cam_node)
            scene.remove_node(light_node)

        renderer.delete()
        return rendered_paths

    except Exception as e:
        warnings.warn(f"Rendering failed: {e}")
        return []