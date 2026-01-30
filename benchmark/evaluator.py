import csv
from utils.io import load_mesh
from utils.safe_exec import safe_metric
from metrics.mesh_stats import mesh_stats
from metrics.integrity import mesh_integrity
from metrics.chamfer import chamfer_distance

def evaluate(mesh_path, ref_path=None):
    mesh = load_mesh(mesh_path)
    results = {}

    results.update(safe_metric(mesh_stats, mesh, default={}))
    results.update(safe_metric(mesh_integrity, mesh, default={}))

    if ref_path:
        ref = load_mesh(ref_path)
        results["chamfer"] = safe_metric(
            chamfer_distance, mesh, ref, default=None
        )

    return results


def run_batch(mesh_paths, out_csv):
    fieldnames = set()
    rows = []

    for path in mesh_paths:
        res = evaluate(path)
        res["mesh"] = path
        rows.append(res)
        fieldnames |= res.keys()

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
