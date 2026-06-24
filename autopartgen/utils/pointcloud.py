# Copyright (c) Meta Platforms, Inc. and affiliates.

import fpsample
import numpy as np
import torch


def gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather batched points with indices shaped ``(B, K)``."""
    if indices.device != points.device:
        indices = indices.to(points.device)
    expand_shape = indices.shape + points.shape[2:]
    gather_idx = indices.reshape(indices.shape[0], indices.shape[1], *([1] * (points.ndim - 2)))
    gather_idx = gather_idx.expand(expand_shape)
    return torch.gather(points, dim=1, index=gather_idx)


def _sample_farthest_points(pc, K, random_start_point=True, sampling_type="fps"):
    assert sampling_type in ["fps", "fps_full"]
    assert pc.ndim == 3

    # Use fpsample for both CPU and CUDA tensors. Moving 32K points to CPU is a
    # small cost here and avoids a hard PyTorch3D dependency in the release env.
    assert pc.shape[0] == 1, "fpsample path supports batch size 1"
    start_idx = (
        int(torch.randint(pc.shape[1], (1,), device="cpu").item())
        if random_start_point
        else 0
    )
    pc_in = pc[0, :, :3] if sampling_type == "fps" else pc[0]
    pc_np = pc_in.detach().to("cpu", dtype=torch.float32).numpy()
    level = 5 if pc_np.shape[0] <= 25_000 else 7
    indices_np = fpsample.bucket_fps_kdline_sampling(pc_np, K, level, start_idx=start_idx)
    indices = torch.from_numpy(indices_np)[None].long().to(pc.device)
    pc_out = gather_points(pc, indices)

    return pc_out, indices


def sample_pc(
    points,
    N,
    sampling_type="random",
    fps_max_points=None,
    fps_random=True,
    fps_chunks=1,
    return_ind=False,
):
    assert isinstance(points, torch.Tensor)
    assert points.ndim == 3
    assert N % fps_chunks == 0
    B, P, C = points.shape

    if P <= N:
        if return_ind:
            indices = torch.arange(P, device=points.device).unsqueeze(0).expand(B, -1)
            return points, indices
        return points

    if sampling_type == "random":
        indices = torch.randperm(P, device=points.device)[:N]
        points = points[:, indices]
        if return_ind:
            return points, indices.unsqueeze(0).expand(B, -1)
        return points

    elif sampling_type.startswith("fps"):
        if fps_max_points is not None:
            # Subsample the point set first
            n_fps = max(fps_max_points, N)

            ind = torch.randperm(P, device=points.device)[:n_fps]
            subsampled_points = points[:, ind]  # shape: (B, n_fps, C)
        else:
            n_fps = P
            ind = torch.arange(P, device=points.device)
            subsampled_points = points

        points_list = []
        indices_list = []
        N_by_chunks = N // fps_chunks

        for chunk_id, pc in enumerate(subsampled_points.chunk(fps_chunks, dim=1)):
            pc_out, local_idx = _sample_farthest_points(
                pc,
                K=N_by_chunks,
                random_start_point=fps_random,
                sampling_type=sampling_type,
            )
            points_list.append(pc_out)

            # local_idx is index w.r.t. chunk
            chunk_size = n_fps // fps_chunks
            global_in_subsampled = (
                local_idx + chunk_id * chunk_size
            )  # index into subsampled_points
            global_in_original = ind[
                global_in_subsampled
            ]  # index into full original input
            indices_list.append(global_in_original)

        sampled_points = torch.cat(points_list, dim=1)  # shape: (B, N, C)
        final_indices = torch.cat(indices_list, dim=1)  # shape: (B, N)

        if return_ind:
            return sampled_points, final_indices
        return sampled_points

    else:
        raise NotImplementedError(f"Unsupported sampling_type: {sampling_type}")


def random_rotation_matrix(canonical_axis_only=True, np_rng=None):
    """
    Create a random rotation matrix.
    """
    if np_rng is None:
        np_rng = np.random.default_rng()
    if canonical_axis_only:
        angle = np_rng.choice([0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
        axis = np_rng.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        angle = np_rng.random() * 2 * np.pi
        axis = np_rng.normal(size=3)
        axis /= np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    R = np.array(
        [
            [
                cos_angle + axis[0] ** 2 * (1 - cos_angle),
                axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle,
                axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle,
            ],
            [
                axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle,
                cos_angle + axis[1] ** 2 * (1 - cos_angle),
                axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle,
            ],
            [
                axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle,
                axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle,
                cos_angle + axis[2] ** 2 * (1 - cos_angle),
            ],
        ]
    )
    return R


def random_mirror_matrix():
    """
    Create a random mirror matrix.
    """
    if np.random.rand() < 0.75:
        axis = np.random.choice([0, 1, 2], size=1)[0]
        M = np.eye(3)
        M[axis, axis] = -1
    else:
        M = np.eye(3)
    return M


def apply_transformation(points, normals, transform):
    """
    Apply a transformation matrix to points and normals.
    """
    to_torch_tensor = False
    if isinstance(points, torch.Tensor):
        to_torch_tensor = True
        dtype = points.dtype
        points = points.cpu().numpy()
        normals = normals.cpu().numpy() if normals is not None else None

    transformed_points = np.dot(points, transform.T)

    if normals is not None:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)

        epsilon = 1e-6
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1

        normals /= norms

        transformed_normals = np.dot(normals, transform.T)
        norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)

        epsilon = 1e-6
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1

        transformed_normals /= norms
    else:
        transformed_normals = None

    if to_torch_tensor:
        transformed_points = torch.from_numpy(transformed_points).to(dtype)
        transformed_normals = (
            torch.from_numpy(transformed_normals).to(dtype)
            if transformed_normals is not None
            else None
        )
    return transformed_points, transformed_normals


def canonical_rotation_matrices():
    """Returns a batch of rotation matrices for the 24 canonical orientations."""
    azim = torch.tensor([0] * 4 + [90] * 4 + [180] * 4 + [270] * 4 + [0] * 4 + [90] * 4)
    azim = azim * np.pi / 180
    elev = torch.tensor([0] * 16 + [90] * 2 + [-90] * 2 + [90] * 2 + [-90] * 2)
    elev = elev * np.pi / 180
    roll = torch.tensor([0, 90, 180, 270] * 4 + [0, 90] * 4)
    roll = roll * np.pi / 180
    return _euler_xyz_to_matrix(torch.stack((azim, elev, roll), dim=-1))


def _euler_xyz_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    x, y, z = angles.unbind(-1)
    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return rx @ ry @ rz
