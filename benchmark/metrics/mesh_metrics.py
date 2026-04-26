#!/usr/bin/env python3
"""
Mesh quality metrics for text-to-3D evaluation
"""

import numpy as np
import trimesh
from typing import Dict, Optional
import warnings

from utils.io import load_mesh

class MeshMetrics:
    """Compute geometric and topological metrics for 3D meshes"""
    
    def __init__(self, mesh: trimesh.Trimesh, verbose: bool = False):
        """
        Initialize metrics calculator
        
        Args:
            mesh: Trimesh object to evaluate
            verbose: Print detailed information
        """
        self.mesh = mesh
        self.verbose = verbose
        
    def compute_all_metrics(self, reference_mesh: Optional[trimesh.Trimesh] = None) -> Dict[str, float]:
        """
        Compute all available metrics
        
        Args:
            reference_mesh: Optional reference mesh for comparison metrics
            
        Returns:
            Dictionary of metric_name -> value
        """
        metrics = {}
        
        # Basic geometry metrics
        metrics.update(self._compute_basic_metrics())
        
        # Topology metrics
        metrics.update(self._compute_topology_metrics())
        
        # Bounding box metrics
        metrics.update(self._compute_bbox_metrics())
        
        # Mesh regularity metrics
        metrics.update(self._compute_regularity_metrics())

        # Comparison metrics (if reference provided)
        if reference_mesh is not None:
            metrics.update(self._compute_comparison_metrics(reference_mesh))
            
        return metrics
    
    def _compute_basic_metrics(self) -> Dict[str, float]:
        """Compute basic geometric properties"""
        metrics = {}
        
        try:
            metrics['num_vertices'] = len(self.mesh.vertices)
            metrics['num_faces'] = len(self.mesh.faces)
            
            # Surface area
            metrics['surface_area'] = float(self.mesh.area)
            
            # Volume (only meaningful if watertight)
            try:
                metrics['volume'] = float(self.mesh.volume)
            except:
                metrics['volume'] = 0.0
                if self.verbose:
                    warnings.warn("Volume calculation failed (mesh may not be watertight)")
                    
        except Exception as e:
            if self.verbose:
                print(f"Error in basic metrics: {e}")
            metrics['num_vertices'] = 0
            metrics['num_faces'] = 0
            metrics['surface_area'] = 0.0
            metrics['volume'] = 0.0
            
        return metrics
    
    def _compute_topology_metrics(self) -> Dict[str, float]:
        """Compute topological quality metrics"""
        metrics = {}
        
        try:
            # Watertightness
            metrics['is_watertight'] = float(self.mesh.is_watertight)
            
            # Connected components
            components = self.mesh.split(only_watertight=False)
            metrics['num_components'] = len(components)
            
            # Non-manifold edges (edges shared by more than 2 faces)
            edge_face_count = np.bincount(self.mesh.edges_unique_inverse)
            non_manifold = np.sum(edge_face_count > 2)
            metrics['num_nonmanifold_edges'] = int(non_manifold)
            
            # Normal consistency
            metrics['normal_consistency'] = self._compute_normal_consistency()
            
        except Exception as e:
            if self.verbose:
                print(f"Error in topology metrics: {e}")
            metrics['is_watertight'] = 0.0
            metrics['num_components'] = 0
            metrics['num_nonmanifold_edges'] = 0
            metrics['normal_consistency'] = 0.0
            
        return metrics
    
    def _compute_normal_consistency(self) -> float:
        """
        Measure how consistently normals are oriented
        Returns value between 0 (inconsistent) and 1 (perfectly consistent)
        """
        try:
            # Get face normals
            normals = self.mesh.face_normals
            
            # Get face adjacency
            face_adjacency = self.mesh.face_adjacency
            
            if len(face_adjacency) == 0:
                return 0.0
            
            # Check alignment between adjacent faces
            consistent_count = 0
            for i, j in face_adjacency:
                # Dot product of adjacent normals (should be positive if consistent)
                dot = np.dot(normals[i], normals[j])
                if dot > 0:
                    consistent_count += 1
                    
            consistency = consistent_count / len(face_adjacency)
            return float(consistency)
            
        except Exception as e:
            if self.verbose:
                print(f"Error computing normal consistency: {e}")
            return 0.0
    
    def _compute_bbox_metrics(self) -> Dict[str, float]:
        """Compute bounding box metrics"""
        metrics = {}
        
        try:
            extents = self.mesh.extents
            
            metrics['bbox_x'] = float(extents[0])
            metrics['bbox_y'] = float(extents[1])
            metrics['bbox_z'] = float(extents[2])
            metrics['bbox_volume'] = float(np.prod(extents))
            
        except Exception as e:
            if self.verbose:
                print(f"Error in bbox metrics: {e}")
            metrics['bbox_x'] = 0.0
            metrics['bbox_y'] = 0.0
            metrics['bbox_z'] = 0.0
            metrics['bbox_volume'] = 0.0
            
        return metrics
    
    def _compute_regularity_metrics(self) -> Dict[str, float]:
        """
        Compute mesh regularity / tessellation quality metrics.

        face_aspect_ratio_mean  – mean of (longest_edge / shortest_edge) per triangle;
                                   1.0 = perfectly equilateral, higher = more degenerate.
        face_aspect_ratio_max   – worst-case aspect ratio in the mesh.
        face_area_cv            – coefficient of variation of face areas (std/mean);
                                   0.0 = perfectly uniform, higher = uneven tessellation.
        edge_length_cv          – coefficient of variation of all edge lengths.
        """
        metrics = {}
        try:
            v = self.mesh.vertices
            f = self.mesh.faces

            e0 = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
            e1 = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)
            e2 = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)
            edges = np.stack([e0, e1, e2], axis=1)   # (n_faces, 3)

            max_e = edges.max(axis=1)
            min_e = edges.min(axis=1)
            valid = min_e > 0
            aspect = np.ones(len(f))
            aspect[valid] = max_e[valid] / min_e[valid]

            metrics['face_aspect_ratio_mean'] = float(np.mean(aspect))
            metrics['face_aspect_ratio_max']  = float(np.max(aspect))

            face_areas = self.mesh.area_faces
            area_mean = float(np.mean(face_areas))
            metrics['face_area_cv'] = float(np.std(face_areas) / area_mean) if area_mean > 0 else 0.0

            all_e = edges.flatten()
            e_mean = float(np.mean(all_e))
            metrics['edge_length_cv'] = float(np.std(all_e) / e_mean) if e_mean > 0 else 0.0

        except Exception as e:
            if self.verbose:
                print(f"Error in regularity metrics: {e}")
            metrics['face_aspect_ratio_mean'] = 0.0
            metrics['face_aspect_ratio_max']  = 0.0
            metrics['face_area_cv']           = 0.0
            metrics['edge_length_cv']         = 0.0

        return metrics

    def _compute_comparison_metrics(self, reference: trimesh.Trimesh) -> Dict[str, float]:
        """
        Compute metrics comparing to reference mesh
        
        Args:
            reference: Ground truth mesh
            
        Returns:
            Dictionary with comparison metrics
        """
        metrics = {}
        
        try:
            metrics['chamfer_distance'] = self._compute_chamfer_distance(reference)
        except Exception as e:
            if self.verbose:
                print(f"Error computing Chamfer distance: {e}")
            metrics['chamfer_distance'] = float('inf')
            
        return metrics
    
    def _compute_chamfer_distance(self, reference: trimesh.Trimesh, 
                                   num_samples: int = 10000) -> float:
        """
        Compute Chamfer Distance between two meshes
        
        Args:
            reference: Reference mesh
            num_samples: Number of points to sample from each mesh
            
        Returns:
            Chamfer distance (lower is better)
        """
        # Sample points from both meshes
        points_generated = self.mesh.sample(num_samples)
        points_reference = reference.sample(num_samples)
        
        # Compute nearest neighbor distances in both directions
        from scipy.spatial import cKDTree
        
        # Build KD-trees
        tree_generated = cKDTree(points_generated)
        tree_reference = cKDTree(points_reference)
        
        # Query nearest neighbors
        dist_to_ref, _ = tree_generated.query(points_reference)
        dist_to_gen, _ = tree_reference.query(points_generated)
        
        # Chamfer distance is mean of both directions
        chamfer = (np.mean(dist_to_ref) + np.mean(dist_to_gen)) / 2.0
        
        return float(chamfer)


def safe_compute_metrics(mesh_path: str, 
                        reference_path: Optional[str] = None,
                        verbose: bool = False) -> Dict[str, float]:
    """
    Safely compute metrics with error handling
    
    Args:
        mesh_path: Path to mesh file (.ply, .obj, etc.)
        reference_path: Optional path to reference mesh
        verbose: Print debug information
        
    Returns:
        Dictionary of metrics (with sensible defaults on failure)
    """
    default_metrics = {
        'num_vertices': 0,
        'num_faces': 0,
        'surface_area': 0.0,
        'volume': 0.0,
        'is_watertight': 0.0,
        'num_components': 0,
        'num_nonmanifold_edges': 0,
        'normal_consistency': 0.0,
        'bbox_x': 0.0,
        'bbox_y': 0.0,
        'bbox_z': 0.0,
        'bbox_volume': 0.0,
        'face_aspect_ratio_mean': 0.0,
        'face_aspect_ratio_max':  0.0,
        'face_area_cv':           0.0,
        'edge_length_cv':         0.0,
        'load_success': 0.0,
        'error_message': ''
    }
    
    try:
        # Load mesh (with degenerate/duplicate face cleanup)
        mesh = load_mesh(mesh_path)
        default_metrics['load_success'] = 1.0

        # Load reference if provided
        reference = None
        if reference_path is not None:
            try:
                reference = load_mesh(reference_path)
            except Exception as e:
                if verbose:
                    print(f"Failed to load reference mesh: {e}")
        
        # Compute metrics
        calculator = MeshMetrics(mesh, verbose=verbose)
        metrics = calculator.compute_all_metrics(reference)
        
        # Merge with defaults
        default_metrics.update(metrics)
        default_metrics['error_message'] = 'success'
        
    except Exception as e:
        default_metrics['error_message'] = str(e)
        if verbose:
            print(f"Error processing {mesh_path}: {e}")
    
    return default_metrics