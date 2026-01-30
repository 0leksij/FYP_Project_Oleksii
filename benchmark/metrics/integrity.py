import numpy as np

def mesh_integrity(mesh):
    return {
        "is_watertight": bool(mesh.is_watertight),
        "num_components": int(len(mesh.split())),
        "num_nonmanifold_edges": int(
            len(mesh.edges_unique) - len(mesh.edges_unique_face)
        ),
        "normal_consistency": float(mesh.face_normals_consistent)
    }

def bounding_box_stats(mesh):
    extents = mesh.bounding_box.extents
    return {
        "bbox_x": float(extents[0]),
        "bbox_y": float(extents[1]),
        "bbox_z": float(extents[2]),
        "bbox_volume": float(extents.prod())
    }
