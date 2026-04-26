import trimesh

def load_mesh(path):
    # process=True (default) removes degenerate/duplicate faces and unreferenced vertices
    mesh = trimesh.load(path, force='mesh', process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Not a valid mesh")
    return mesh
