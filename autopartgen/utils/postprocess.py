# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Scene-level part post-processing.

The cleanup splits decoded parts into disconnected components, then drops
components that are not connected to the grounded set. This removes detached
surface fragments while preserving the intended part boundaries.
"""

from typing import List

import numpy as np
import trimesh

try:  # Use GPU knn when available; fall back to scipy on CPU-only setups.
    import torch
    from pytorch3d.ops import knn_points

    _HAS_P3D = True
except Exception:  # pragma: no cover
    _HAS_P3D = False


def split_by_loose_parts(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    """Split a mesh into disconnected components (Blender 'Separate by Loose Parts')."""
    parts = mesh.split(only_watertight=False)
    for p in parts:
        p.remove_unreferenced_vertices()
    return list(parts)


def _bbox_distance(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    ca, ea = a.bounding_box.centroid, a.bounding_box.extents
    cb, eb = b.bounding_box.centroid, b.bounding_box.extents
    d = np.maximum(0.0, np.abs(cb - ca) - ea / 2 - eb / 2)
    return float(np.linalg.norm(d))


def part_distance(a: trimesh.Trimesh, b: trimesh.Trimesh, n_samples: int = 10000) -> float:
    """Minimum surface-to-surface distance between two parts (knn over sampled points)."""
    try:
        pa, _ = trimesh.sample.sample_surface(a, n_samples)
        pb, _ = trimesh.sample.sample_surface(b, n_samples)
    except Exception:
        return _bbox_distance(a, b)
    if len(pa) == 0 or len(pb) == 0:
        return _bbox_distance(a, b)
    if _HAS_P3D:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ta = torch.tensor(np.asarray(pa), dtype=torch.float32, device=dev)[None]
        tb = torch.tensor(np.asarray(pb), dtype=torch.float32, device=dev)[None]
        d, _, _ = knn_points(ta, tb, K=1)
        return float(d.min().item())
    from scipy.spatial import cKDTree

    return float(cKDTree(np.asarray(pb)).query(np.asarray(pa), k=1)[0].min())


def ground_connectivity_check(
    parts: List[trimesh.Trimesh], connection_distance: float = 0.001
) -> List[trimesh.Trimesh]:
    """Flood-fill from the first part; drop components farther than the threshold.

    The first part is the ground reference; any part within ``connection_distance``
    of the grounded set becomes grounded (transitively). Detached floaters / comb
    fragments are removed.
    """
    if not parts:
        return parts
    ground = [parts[0]]
    floaters = list(parts[1:])
    while True:
        ground_mesh = ground[0] if len(ground) == 1 else trimesh.util.concatenate(ground)
        if len(ground) > 1:
            ground_mesh.remove_unreferenced_vertices()
        newly, remaining = [], []
        for f in floaters:
            if part_distance(f, ground_mesh) <= connection_distance:
                newly.append(f)
            else:
                remaining.append(f)
        if not newly:
            break
        ground.extend(newly)
        floaters = remaining
    return ground


def _ground_connected_entries(
    entries: List[tuple[int, trimesh.Trimesh]], connection_distance: float = 0.001
) -> List[tuple[int, trimesh.Trimesh]]:
    if not entries:
        return entries
    ground = [entries[0]]
    floaters = list(entries[1:])
    while True:
        ground_meshes = [m for _, m in ground]
        ground_mesh = (
            ground_meshes[0]
            if len(ground_meshes) == 1
            else trimesh.util.concatenate(ground_meshes)
        )
        if len(ground_meshes) > 1:
            ground_mesh.remove_unreferenced_vertices()
        newly, remaining = [], []
        for entry in floaters:
            _, mesh = entry
            if part_distance(mesh, ground_mesh) <= connection_distance:
                newly.append(entry)
            else:
                remaining.append(entry)
        if not newly:
            break
        ground.extend(newly)
        floaters = remaining
    return ground


def postprocess_parts(
    parts: List[trimesh.Trimesh],
    scene_handling: bool = True,
    preserve_groups: bool = False,
) -> List[trimesh.Trimesh]:
    """Split each part into loose components, then drop floating ones.

    No fragment merge is performed; cleanup is split + ground-connectivity check.
    """
    if not parts:
        return []
    if preserve_groups:
        entries: List[tuple[int, trimesh.Trimesh]] = []
        for part_idx, m in enumerate(parts):
            if m is None or len(m.faces) == 0:
                continue
            for comp in split_by_loose_parts(m):
                entries.append((part_idx, comp))
        if len(entries) > 1 and scene_handling:
            entries = _ground_connected_entries(entries, connection_distance=0.001)

        grouped: dict[int, List[trimesh.Trimesh]] = {}
        for part_idx, comp in entries:
            grouped.setdefault(part_idx, []).append(comp)

        kept_parts: List[trimesh.Trimesh] = []
        for part_idx in sorted(grouped):
            comps = grouped[part_idx]
            if len(comps) == 1:
                kept_parts.append(comps[0])
            else:
                merged = trimesh.util.concatenate(comps)
                merged.remove_unreferenced_vertices()
                kept_parts.append(merged)
        return kept_parts

    cleaned: List[trimesh.Trimesh] = []
    for m in parts:
        if m is None or len(m.faces) == 0:
            continue
        cleaned.extend(split_by_loose_parts(m))
    if len(cleaned) > 1 and scene_handling:
        cleaned = ground_connectivity_check(cleaned, connection_distance=0.001)
    return cleaned
