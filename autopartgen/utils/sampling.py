# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Inference helpers for the autoregressive part-generation pipeline.

Latent encoding with the VAE scale factor, surface resampling from meshes, the
end-of-sequence (EoT) check, and voxel-IoU part deduplication.
"""

from typing import List

import numpy as np
import torch
import trimesh

from autopartgen.utils.pointcloud import sample_pc


def encode_surface(
    surface: torch.Tensor,
    vae: torch.nn.Module,
    scale_factor: float,
    sample_posterior: bool = False,
) -> torch.Tensor:
    """Encode a surface point cloud (B, N, 6 = xyz+normals) to a scaled latent.

    Latents enter the diffusion model in SCALED space (native * scale_factor); the
    sampler divides back by the same factor before decoding. Returns (B, num_latents, C).
    """
    dist = vae.encode(surface).latent_dist
    latent = dist.sample() if sample_posterior else dist.mode()
    return latent * scale_factor


def resample_surface_from_mesh(
    meshes: List[trimesh.Trimesh],
    point_cloud_size: int,
    surface_sampling_type: str = "fps",
    fps_max_points: int = 65536,
    device: str = "cuda",
    num_surface_samples: int = 500_000,
) -> torch.Tensor:
    """Combine meshes, sample surface points with normals, and FPS-downsample.

    Returns (1, point_cloud_size, 6) on ``device``; an all-zero tensor if the
    combined mesh has no faces (used as the empty/start-of-sequence history).
    """
    combined = trimesh.util.concatenate(meshes)
    if combined.is_empty or combined.faces is None or len(combined.faces) == 0:
        return torch.zeros(size=(1, point_cloud_size, 6), device=device)

    surface_points, face_index = combined.sample(num_surface_samples, return_index=True)
    surface_normals = combined.face_normals[face_index]
    surface_all = torch.from_numpy(surface_points).float()
    surface_normals_t = torch.from_numpy(surface_normals).float()
    surface_all = torch.cat([surface_all, surface_normals_t], dim=-1)
    return sample_pc(
        surface_all[None],
        point_cloud_size,
        sampling_type=surface_sampling_type,
        fps_max_points=fps_max_points,
    ).to(device)


def is_near_zero_output(tensor: torch.Tensor, threshold: float = 0.1) -> bool:
    """True if a sampled latent is near-zero (the end-of-sequence / EoT token)."""
    return bool(torch.abs(tensor).mean() < threshold)


class IoUCalculator:
    """Voxel-based IoU for detecting near-duplicate parts.

    Open3D is used when installed. A sampled-surface voxel fallback keeps the
    release usable in smaller environments.
    """

    def __init__(self, grid_size: int = 256, fallback_samples: int = 500_000) -> None:
        if int(grid_size) <= 0:
            raise ValueError(f"grid_size must be a positive integer, got {grid_size}")
        self.grid_size = int(grid_size)
        self.voxel_size = 2.0 / self.grid_size
        self.min_bound = np.array([-1.0, -1.0, -1.0])
        self.max_bound = np.array([1.0, 1.0, 1.0])
        self.fallback_samples = int(fallback_samples)
        try:
            import open3d as o3d  # type: ignore
        except Exception:
            o3d = None
        self._o3d = o3d

    def _voxels(self, mesh: trimesh.Trimesh) -> set:
        if mesh.is_empty or len(mesh.faces) == 0:
            return set()
        if self._o3d is None:
            return self._sampled_surface_voxels(mesh)
        o3d = self._o3d
        mesh_o3d = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices),
            triangles=o3d.utility.Vector3iVector(mesh.faces),
        )
        grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            input=mesh_o3d,
            voxel_size=self.voxel_size,
            min_bound=self.min_bound,
            max_bound=self.max_bound,
        )
        return {tuple(v.grid_index) for v in grid.get_voxels()}

    def _sampled_surface_voxels(self, mesh: trimesh.Trimesh) -> set:
        n = max(1024, min(self.fallback_samples, max(1024, len(mesh.faces) * 8)))
        points = mesh.sample(n)
        idx = np.floor((np.clip(points, -1.0, 1.0) + 1.0) * (self.grid_size / 2.0))
        idx = np.clip(idx.astype(np.int64), 0, self.grid_size - 1)
        keys = (
            idx[:, 0] * self.grid_size * self.grid_size
            + idx[:, 1] * self.grid_size
            + idx[:, 2]
        )
        return set(np.unique(keys).tolist())

    def compute_iou(self, mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> float:
        v1, v2 = self._voxels(mesh1), self._voxels(mesh2)
        if not v1 or not v2:
            return 0.0
        inter = len(v1 & v2)
        return float(max(inter / len(v1), inter / len(v2)))
