import trimesh

def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Not a valid mesh")
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh
