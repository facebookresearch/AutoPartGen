# Copyright (c) Meta Platforms, Inc. and affiliates.

import colorsys

import logging
import os
import warnings
from typing import Optional, Sequence

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes as sk_marching_cubes

from .io import makedirs

from .utils3d_torch import (
    compute_connected_components,
    compute_dual_graph,
    compute_edge_connected_components,
    compute_edges,
    remove_unreferenced_vertices,
    sphere_hammersley_sequence,
)


LOGGER = logging.getLogger(__name__)

MAX_N_FACES = 100_000
DEFAULT_SIMPLIFY_FACES = 50_000
# Drop disconnected components whose surface area is below this fraction of the
# part's total area.
FLOATER_REMOVAL_RATIO = float(os.environ.get("APG_FLOAT_RATIO", "0.01"))
SIMPLIFY_AGG = float(os.environ.get("APG_SIMPLIFY_AGG", "1.0"))
HIDDEN_REMOVAL_MAX_N_FACES = 400_000  # too slow for large meshes


def generate_distinct_colors(
    n: int, saturation: float = 0.7, value: float = 0.95
) -> np.ndarray:
    """Generate n distinct RGBA colors as uint8."""
    if n <= 0:
        return np.zeros((0, 4), dtype=np.uint8)
    colors = []
    for i in range(n):
        h = (i / float(n)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255), 255))
    return np.asarray(colors, dtype=np.uint8)


_generate_distinct_colors = generate_distinct_colors


def _ensure_output_dir(path: str) -> str:
    """Ensure the directory for a file path or a dir path exists."""
    lower = path.lower()
    if lower.endswith((".obj", ".ply", ".stl", ".glb", ".gltf")):
        dir_path = os.path.dirname(path) or "."
    else:
        dir_path = path
    makedirs(dir_path, exist_ok=True)
    return dir_path


def combine_meshes_with_colors(
    meshes: Sequence[trimesh.Trimesh],
    colors: Optional[Sequence[Sequence[int]]] = None,
) -> trimesh.Trimesh:
    """
    Combine a sequence of meshes into a single mesh, assigning a distinct color per part.
    Colors are applied as per-vertex RGBA (uint8) so they are preserved in common formats like PLY/OBJ.
    """
    if meshes is None or len(meshes) == 0:
        raise ValueError("meshes must be a non-empty sequence of trimesh.Trimesh")
    for m in meshes:
        if not isinstance(m, trimesh.Trimesh):
            raise TypeError("All items in meshes must be trimesh.Trimesh")

    # Drop empty parts: a single empty mesh in trimesh.util.concatenate makes it
    # discard all per-vertex colors, collapsing the combined mesh to one color.
    meshes = [m for m in meshes if len(m.vertices) > 0 and len(m.faces) > 0]
    if len(meshes) == 0:
        raise ValueError("all meshes are empty")

    num_parts = len(meshes)
    if colors is None:
        color_palette = _generate_distinct_colors(num_parts)
    else:
        color_palette = np.asarray(colors, dtype=np.uint8)
        if color_palette.shape[1] == 3:
            # add alpha if missing
            alpha = 255 * np.ones((color_palette.shape[0], 1), dtype=np.uint8)
            color_palette = np.concatenate([color_palette, alpha], axis=1)

    colored_parts: list[trimesh.Trimesh] = []
    for idx, mesh in enumerate(meshes):
        part = mesh.copy()
        col = color_palette[idx % len(color_palette)]
        vcolors = np.tile(col[None, :], (len(part.vertices), 1))
        part.visual.vertex_colors = vcolors
        colored_parts.append(part)

    combined = trimesh.util.concatenate(colored_parts)
    return combined


def normalize_trimesh(mesh, scale=1.0, ord=None):
    # center and unit normalize (modulo scale)
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError("Cannot normalize an empty mesh (no vertices).")
    center = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2
    vertices = mesh.vertices - center
    max_norm = np.linalg.norm(vertices, axis=1, ord=ord).max()
    if max_norm <= 0:
        raise ValueError("Cannot normalize a degenerate mesh (zero extent).")
    vertices = vertices / max_norm
    mesh.vertices = scale * vertices

    return mesh


def normalize_pytorch3d(
    meshes, center=True, scale_mode="unit_cube", inplace=False, use_center_mass=False
):
    if center:
        if use_center_mass:
            from pytorch3d.ops import sample_points_from_meshes as sample_points

            offsets = sample_points(meshes, 100000).mean(1)
        else:
            offsets = 0.5 * (
                meshes.verts_padded().max(1)[0] + meshes.verts_padded().min(1)[0]
            )
        # meshes.offset_vert requires tensor of size (all_V, 3), while offsets is (B, 3)
        NVs = meshes.num_verts_per_mesh()
        offsets = torch.cat(
            [offset[None].expand(nv, -1) for offset, nv in zip(offsets, NVs)], dim=0
        )
        meshes = (
            meshes.offset_verts_(-offsets) if inplace else meshes.offset_verts(-offsets)
        )

    if scale_mode == "none" or scale_mode is None:
        scales = 1.0
    elif scale_mode == "unit_cube":
        scales = meshes.verts_padded().abs().flatten(1).max(1)[0] * 2  # [-0.5, 0.5]^3
    elif scale_mode == "unit_sphere":
        scales = meshes.verts_padded().norm(dim=2).max(1)[0] * 2
    else:
        raise NotImplementedError
    return (
        meshes.scale_verts_(1 / scales) if inplace else meshes.scale_verts(1 / scales)
    )


def generate_meshgrid(grid_size=512):
    vol_axes = [
        torch.linspace(-1.0, 1.0, s, dtype=torch.float) for s in [grid_size] * 3
    ]
    grid = torch.stack(torch.meshgrid(*vol_axes, indexing="ij"), dim=-1).view(-1, 3)[None]
    return grid


def generate_dense_grid_points_gpu(
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    octree_depth: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = 2**octree_depth
    device = bbox_min.device

    x = torch.linspace(
        bbox_min[0], bbox_max[0], int(num_cells), dtype=torch.float32, device=device
    )
    y = torch.linspace(
        bbox_min[1], bbox_max[1], int(num_cells), dtype=torch.float32, device=device
    )
    z = torch.linspace(
        bbox_min[2], bbox_max[2], int(num_cells), dtype=torch.float32, device=device
    )

    xs, ys, zs = torch.meshgrid(x, y, z, indexing=indexing)
    xyz = torch.stack((xs, ys, zs), dim=-1)
    xyz = xyz.view(-1, 3)
    grid_size = [int(num_cells), int(num_cells), int(num_cells)]

    return xyz, grid_size, length


def find_mesh_grid_coordinates_fast_gpu(occupancy_grid, n_limits=-1):
    core_grid = occupancy_grid[1:-1, 1:-1, 1:-1]
    occupied = core_grid > 0

    neighbors_unoccupied = (
        (occupancy_grid[:-2, :-2, :-2] < 0)
        | (occupancy_grid[:-2, :-2, 1:-1] < 0)
        | (occupancy_grid[:-2, :-2, 2:] < 0)  # x-1, y-1, z-1/0/1
        | (occupancy_grid[:-2, 1:-1, :-2] < 0)
        | (occupancy_grid[:-2, 1:-1, 1:-1] < 0)
        | (occupancy_grid[:-2, 1:-1, 2:] < 0)  # x-1, y0, z-1/0/1
        | (occupancy_grid[:-2, 2:, :-2] < 0)
        | (occupancy_grid[:-2, 2:, 1:-1] < 0)
        | (occupancy_grid[:-2, 2:, 2:] < 0)  # x-1, y+1, z-1/0/1
        | (occupancy_grid[1:-1, :-2, :-2] < 0)
        | (occupancy_grid[1:-1, :-2, 1:-1] < 0)
        | (occupancy_grid[1:-1, :-2, 2:] < 0)  # x0, y-1, z-1/0/1
        | (occupancy_grid[1:-1, 1:-1, :-2] < 0)
        | (occupancy_grid[1:-1, 1:-1, 2:] < 0)  # x0, y0, z-1/1
        | (occupancy_grid[1:-1, 2:, :-2] < 0)
        | (occupancy_grid[1:-1, 2:, 1:-1] < 0)
        | (occupancy_grid[1:-1, 2:, 2:] < 0)  # x0, y+1, z-1/0/1
        | (occupancy_grid[2:, :-2, :-2] < 0)
        | (occupancy_grid[2:, :-2, 1:-1] < 0)
        | (occupancy_grid[2:, :-2, 2:] < 0)  # x+1, y-1, z-1/0/1
        | (occupancy_grid[2:, 1:-1, :-2] < 0)
        | (occupancy_grid[2:, 1:-1, 1:-1] < 0)
        | (occupancy_grid[2:, 1:-1, 2:] < 0)  # x+1, y0, z-1/0/1
        | (occupancy_grid[2:, 2:, :-2] < 0)
        | (occupancy_grid[2:, 2:, 1:-1] < 0)
        | (occupancy_grid[2:, 2:, 2:] < 0)  # x+1, y+1, z-1/0/1
    )
    core_mesh_coords = (
        torch.nonzero(occupied & neighbors_unoccupied, as_tuple=False) + 1
    )

    if n_limits != -1 and core_mesh_coords.shape[0] > n_limits:
        LOGGER.debug(
            "core mesh coords %d too large, limited to %d",
            core_mesh_coords.shape[0], n_limits,
        )
        ind = np.random.choice(core_mesh_coords.shape[0], n_limits, True)
        core_mesh_coords = core_mesh_coords[ind]

    return core_mesh_coords


def find_candidates_band(
    occupancy_grid: torch.Tensor,
    band_threshold: float,
    n_limits: int = -1,
    apply_sigmoid: bool = True,
) -> torch.Tensor:
    """
    Returns the coordinates of all voxels in the occupancy_grid where |value| < band_threshold.

    Args:
        occupancy_grid (torch.Tensor): A 3D tensor of SDF values.
        band_threshold (float): The threshold below which |SDF| must be to include the voxel.
        n_limits (int): Maximum number of points to return (-1 for no limit)

    Returns:
        torch.Tensor: A 2D tensor of coordinates (N x 3) where each row is [x, y, z].
    """
    core_grid = occupancy_grid[1:-1, 1:-1, 1:-1]
    # Match upstream TripoSG: band in bounded SDF space sigmoid(x)*2-1, not on raw
    # (unbounded, steep) logits -- raw banding gives a too-thin, mis-shaped shell so
    # coarse values survive near the surface and produce spurious zero-crossings
    # (floaters). The coarse ROI must pass apply_sigmoid=False (sigmoid |.|<1.0 would
    # select every voxel and disable the ROI crop).
    if apply_sigmoid:
        core_grid = torch.sigmoid(core_grid) * 2 - 1
    # Create a boolean mask for all cells in the band
    in_band = torch.abs(core_grid) < band_threshold

    # Get coordinates of all voxels in the band
    core_mesh_coords = torch.nonzero(in_band, as_tuple=False) + 1

    if n_limits != -1 and core_mesh_coords.shape[0] > n_limits:
        LOGGER.debug(
            "core mesh coords %d too large, limited to %d",
            core_mesh_coords.shape[0], n_limits,
        )
        ind = np.random.choice(core_mesh_coords.shape[0], n_limits, True)
        core_mesh_coords = core_mesh_coords[ind]

    return core_mesh_coords


def expand_edge_region_fast(edge_coords, grid_size):
    expanded_tensor = torch.zeros(
        grid_size,
        grid_size,
        grid_size,
        device=edge_coords.device,
        dtype=torch.float32,
        requires_grad=False,
    )
    expanded_tensor[edge_coords[:, 0], edge_coords[:, 1], edge_coords[:, 2]] = 1
    if grid_size < 512:
        kernel_size = 5
        pooled_tensor = torch.nn.functional.max_pool3d(
            expanded_tensor.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=2,
        ).squeeze()
    else:
        kernel_size = 3
        pooled_tensor = torch.nn.functional.max_pool3d(
            expanded_tensor.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        ).squeeze()
    expanded_coords_low_res = torch.nonzero(pooled_tensor, as_tuple=False).to(
        torch.int16
    )

    expanded_coords_high_res = torch.stack(
        [
            torch.cat(
                (
                    expanded_coords_low_res[:, 0] * 2,
                    expanded_coords_low_res[:, 0] * 2,
                    expanded_coords_low_res[:, 0] * 2,
                    expanded_coords_low_res[:, 0] * 2,
                    expanded_coords_low_res[:, 0] * 2 + 1,
                    expanded_coords_low_res[:, 0] * 2 + 1,
                    expanded_coords_low_res[:, 0] * 2 + 1,
                    expanded_coords_low_res[:, 0] * 2 + 1,
                )
            ),
            torch.cat(
                (
                    expanded_coords_low_res[:, 1] * 2,
                    expanded_coords_low_res[:, 1] * 2,
                    expanded_coords_low_res[:, 1] * 2 + 1,
                    expanded_coords_low_res[:, 1] * 2 + 1,
                    expanded_coords_low_res[:, 1] * 2,
                    expanded_coords_low_res[:, 1] * 2,
                    expanded_coords_low_res[:, 1] * 2 + 1,
                    expanded_coords_low_res[:, 1] * 2 + 1,
                )
            ),
            torch.cat(
                (
                    expanded_coords_low_res[:, 2] * 2,
                    expanded_coords_low_res[:, 2] * 2 + 1,
                    expanded_coords_low_res[:, 2] * 2,
                    expanded_coords_low_res[:, 2] * 2 + 1,
                    expanded_coords_low_res[:, 2] * 2,
                    expanded_coords_low_res[:, 2] * 2 + 1,
                    expanded_coords_low_res[:, 2] * 2,
                    expanded_coords_low_res[:, 2] * 2 + 1,
                )
            ),
        ],
        dim=1,
    )

    return expanded_coords_high_res


def parallel_zoom(occupancy_grid, scale_factor):
    result = torch.nn.functional.interpolate(
        occupancy_grid.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor
    )
    return result.squeeze(0).squeeze(0)


def marching_cubes(voxels, isolevel=0.0, backend="skimage", padding=2):
    """
    assumes ascent gradient direction (inside_values > outside_values)
    """
    device = voxels.device
    for _ in range(padding):
        # we pad voxels to close the volume and avoid boundary holes
        # we do so by extrapolating neighboring values using gradients
        voxels = torch.nn.functional.pad(
            voxels[None], (1, 1, 1, 1, 1, 1), mode="replicate"
        )[0]
        voxels[0, :, :] -= (voxels[2, :, :] - voxels[1, :, :]).abs()
        voxels[-1, :, :] -= (voxels[-3, :, :] - voxels[-2, :, :]).abs()
        voxels[:, 0, :] -= (voxels[:, 2, :] - voxels[:, 1, :]).abs()
        voxels[:, -1, :] -= (voxels[:, -3, :] - voxels[:, -2, :]).abs()
        voxels[:, :, 0] -= (voxels[:, :, 2] - voxels[:, :, 1]).abs()
        voxels[:, :, -1] -= (voxels[:, :, -3] - voxels[:, :, -2]).abs()

    dx = 2.0 / (voxels.shape[-1] - 1)
    rescale_factor = 1.0 / (1.0 - dx * padding)

    if backend == "skimage":
        # 2x slower than GPU-pytorch3d, but more accurate
        verts, faces, _, _ = sk_marching_cubes(
            voxels.cpu().numpy(),
            level=isolevel,
            spacing=(dx, dx, dx),
            gradient_direction="ascent",
            allow_degenerate=False,
        )
        verts -= 1.0

    elif backend == "pytorch3d":
        # 2x faster than skimage, but less accurate and may yield degenerate outputs
        from pytorch3d.ops.marching_cubes import marching_cubes as p3d_marching_cubes

        verts, faces = p3d_marching_cubes(
            voxels.unsqueeze(0), isolevel=0, return_local_coords=True
        )
        verts = verts[0]
        faces = faces[0]
        # TODO: needs to resolve the cases when verts or faces are empty for metric cpu
        if len(verts) == 0:
            verts = torch.empty(0, 3, dtype=torch.float, device=device)
            faces = torch.empty(0, 3, dtype=torch.long, device=device)
        verts = torch.stack([verts[:, 2], verts[:, 1], verts[:, 0]], dim=-1)
        verts = verts.cpu().numpy()
        faces = faces.cpu().numpy()
    else:
        raise NotImplementedError(f"Backend {backend} not implemented")

    verts *= rescale_factor
    return trimesh.Trimesh(verts, faces)


def postprocess(
    mesh,
    floater_removal=True,
    floater_removal_ratio=FLOATER_REMOVAL_RATIO,
    simplification=False,
    simplification_n_faces=MAX_N_FACES,
    smoothing_iters=0,
    hidden_face_removal=True,
    device="cpu",
):
    # Decimate first. On 512/DiffDMC outputs a single part can have millions of
    # faces, and connected-component cleanup is much cheaper after decimation.
    if simplification and len(mesh.faces) > simplification_n_faces:
        mesh = simplify(mesh, simplification_n_faces)
    if floater_removal:
        mesh = remove_floaters(mesh, floater_removal_ratio)
    if smoothing_iters > 0 and len(mesh.faces) > 20:
        # Light Taubin smoothing reduces high-frequency surface noise without
        # shrinking the shape or losing real edges.
        try:
            trimesh.smoothing.filter_taubin(mesh, iterations=smoothing_iters)
        except Exception:
            pass
    return mesh


def remove_floaters(mesh, removal_ratio):
    """Drop small disconnected components below ``removal_ratio`` of total surface AREA.

    Area-ratio cull, not keep-largest-by-face-count. Thin fragments can have many
    tiny triangles but low area, so the area test is the more reliable filter.
    Never returns an empty mesh.
    """
    if mesh is None or len(mesh.faces) == 0 or removal_ratio <= 0:
        return mesh
    comps = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=1,
    )
    if len(comps) <= 1:
        return mesh
    face_areas = mesh.area_faces
    comp_areas = [
        float(face_areas[np.asarray(comp, dtype=np.int64)].sum()) for comp in comps
    ]
    total_area = float(sum(comp_areas))
    if total_area <= 0:
        return mesh
    keep = np.zeros(len(mesh.faces), dtype=bool)
    for comp, area in zip(comps, comp_areas):
        if (area / total_area) >= removal_ratio:
            keep[np.asarray(comp, dtype=np.int64)] = True
    if keep.all() or not keep.any():
        return mesh
    return mesh.submesh([keep], append=True, repair=False)


def simplify(mesh, max_face_count):
    if (
        max_face_count is None
        or max_face_count <= 0
        or len(mesh.faces) <= max_face_count
    ):
        return mesh
    try:
        import fast_simplification

        vertices, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
            target_count=int(max_face_count),
            agg=SIMPLIFY_AGG,
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except Exception:
        pass

    try:
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.add_mesh(
            pymeshlab.Mesh(
                vertex_matrix=np.asarray(mesh.vertices),
                face_matrix=np.asarray(mesh.faces),
            )
        )
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(max_face_count)
        )
        simplified = ms.current_mesh()
        return trimesh.Trimesh(
            vertices=simplified.vertex_matrix(),
            faces=simplified.face_matrix(),
            process=False,
        )
    except Exception:
        pass

    try:
        import open3d

        mesh_o3d = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(mesh.vertices),
            triangles=open3d.utility.Vector3iVector(mesh.faces),
        )
        mesh_o3d = mesh_o3d.remove_duplicated_vertices()
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(max_face_count)
        return trimesh.Trimesh(vertices=mesh_o3d.vertices, faces=mesh_o3d.triangles)
    except Exception:
        if hasattr(mesh, "simplify_quadric_decimation"):
            try:
                return mesh.simplify_quadric_decimation(max_face_count)
            except Exception:
                pass
        warnings.warn(
            "Mesh simplification requested but no simplification backend is available; "
            "returning the unsimplified mesh.",
            RuntimeWarning,
        )
        return mesh


@torch.no_grad()
def remove_hidden_faces(
    mesh,
    max_hole_size=0.04,
    resolution=768,
    num_views=512,
    num_render_batch=32,
    device="cpu",
):
    """
    Source: https://github.com/microsoft/TRELLIS/blob/main/trellis/utils/postprocessing_utils.py

    Rasterize a mesh from multiple views and remove invisible faces.
    Also includes postprocessing to:
        1. Remove connected components that are have low visibility.
        2. Mincut to remove faces at the inner side of the mesh connected to the outer side with a small hole.
    """
    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        MeshRasterizer,
        RasterizationSettings,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Can't import EGL.*")
            from pytorch3d.renderer.opengl import MeshRasterizerOpenGL

        open_gl_backend = True
    except Exception:
        MeshRasterizerOpenGL = None
        open_gl_backend = False

    verts = torch.tensor(mesh.vertices).float().to(device)
    faces = torch.tensor(mesh.faces.astype(np.int32)).to(device)

    ## Rasterize the mesh from multiple views using Pytorch3D
    import igraph

    faces = faces.long()
    mesh = Meshes(verts=[verts], faces=[faces])
    radius = 3.0
    all_yaws = []
    all_pitches = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        all_yaws.append(y)
        all_pitches.append(p)
    all_yaws = torch.tensor(all_yaws, device=device)
    all_pitches = torch.tensor(all_pitches, device=device)

    raster_settings = RasterizationSettings(
        image_size=resolution, blur_radius=0.0, faces_per_pixel=1, bin_size=None
    )

    camXs = torch.sin(all_yaws) * torch.cos(all_pitches) * radius
    camYs = torch.cos(all_yaws) * torch.cos(all_pitches) * radius
    camZs = torch.sin(all_pitches) * radius
    camXYZ = torch.stack([camXs, camYs, camZs], dim=-1).float()
    at = (
        torch.tensor(
            [
                [0, 0, 0],
            ]
        )
        .repeat(num_views, 1)
        .to(device)
    ).float()
    up = (
        torch.tensor(
            [
                [0, 1, 0],
            ]
        )
        .repeat(num_views, 1)
        .to(device)
    ).float()

    R, T = look_at_view_transform(eye=camXYZ, at=at, up=up)
    cameras = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        fov=60,
        znear=0.5,
        zfar=5,
    )
    if open_gl_backend:
        rasterizer = MeshRasterizerOpenGL(
            raster_settings=raster_settings, cameras=cameras
        )
    else:
        rasterizer = MeshRasterizer(raster_settings=raster_settings, cameras=cameras)
    rasterizer.to(device)

    visblity = torch.zeros(faces.shape[0], dtype=torch.float32, device=device)
    for i in range(num_views // num_render_batch):
        cameras = rasterizer.cameras[
            list(range(i * num_render_batch, (i + 1) * num_render_batch))
        ]
        fragments = rasterizer(mesh.extend(num_render_batch), cameras=cameras)

        pix_to_face = fragments.pix_to_face[..., 0]

        valid_mask = pix_to_face >= 0

        for j in range(num_render_batch):
            face_ids = pix_to_face[j][valid_mask[j]]
            face_ids = torch.unique(face_ids)

            visblity[face_ids] += 1.0

    visblity /= num_views

    # Mincut
    ## construct outer faces
    edges, face2edge, edge_degrees = compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    connected_components = compute_connected_components(faces, edges, face2edge)
    outer_face_indices = torch.zeros(
        faces.shape[0], dtype=torch.bool, device=faces.device
    )
    for i in range(len(connected_components)):
        outer_face_indices[connected_components[i]] = visblity[
            connected_components[i]
        ] > min(max(visblity[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)

    ## construct inner faces
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if inner_face_indices.shape[0] == 0:
        vertices, faces = verts.cpu().numpy(), faces.cpu().numpy()
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    ## Construct dual graph (faces as nodes, edges as edges)
    dual_edges, dual_edge2edge = compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_edges_weights = torch.norm(
        verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1
    )

    ## solve mincut problem
    ### construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])
    g.add_edges(dual_edges.cpu().numpy())
    g.es["weight"] = dual_edges_weights.cpu().numpy()

    ### source and target
    g.add_vertex("s")
    g.add_vertex("t")

    ### connect invisible faces to source
    g.add_edges(
        [(f, "s") for f in inner_face_indices],
        attributes={
            "weight": torch.ones(inner_face_indices.shape[0], dtype=torch.float32)
            .cpu()
            .numpy()
        },
    )

    ### connect outer faces to target
    g.add_edges(
        [(f, "t") for f in outer_face_indices],
        attributes={
            "weight": torch.ones(outer_face_indices.shape[0], dtype=torch.float32)
            .cpu()
            .numpy()
        },
    )

    ### solve mincut
    cut = g.mincut("s", "t", (np.array(g.es["weight"]) * 1000).tolist())
    remove_face_indices = torch.tensor(
        [v for v in cut.partition[0] if v < faces.shape[0]],
        dtype=torch.long,
        device=faces.device,
    )

    ### check if the cut is valid with each connected component
    to_remove_cc = compute_connected_components(faces[remove_face_indices])

    valid_remove_cc = []
    for cc in to_remove_cc:
        #### check if the connected component has low visibility
        visblity_max = visblity[remove_face_indices[cc]].max()

        if visblity_max > 0.1:
            continue

        #### check if the cuting loop is small enough
        cc_edge_indices, cc_edges_degree = torch.unique(
            face2edge[remove_face_indices[cc]], return_counts=True
        )
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[
            ~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)
        ]
        if len(cc_new_boundary_edge_indices) > 0:
            cc_new_boundary_edge_cc = compute_edge_connected_components(
                edges[cc_new_boundary_edge_indices]
            )
            cc_new_boundary_edges_cc_center = [
                verts[edges[cc_new_boundary_edge_indices[edge_cc]]]
                .mean(dim=1)
                .mean(dim=0)
                for edge_cc in cc_new_boundary_edge_cc
            ]
            cc_new_boundary_edges_cc_area = []
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = (
                    verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]]
                    - cc_new_boundary_edges_cc_center[i]
                )
                _e2 = (
                    verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]]
                    - cc_new_boundary_edges_cc_center[i]
                )
                cc_new_boundary_edges_cc_area.append(
                    torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5
                )
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue

        valid_remove_cc.append(cc)

    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        faces, verts = remove_unreferenced_vertices(faces, verts)

    vertices, faces = verts.cpu().numpy(), faces.cpu().numpy()
    out_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return out_mesh


def save_colored_mesh(list_of_meshes, out_path: str):
    """
    Color each part with a distinct color and save a combined mesh.

    Always saves GLB files:
    - If out_path is a file path, a single combined GLB is written next to that name (forced .glb extension).
    - If out_path is a directory, one colored GLB per part ("mesh_000.glb", ...) and
      a combined mesh ("mesh_combined.glb") are saved inside it.
    """
    _ensure_output_dir(out_path)

    list_of_meshes = [m for m in list_of_meshes if len(m.vertices) > 0 and len(m.faces) > 0]
    if not list_of_meshes:
        LOGGER.warning("No valid (non-empty) parts to save; nothing written.")
        return

    is_file = out_path.lower().endswith((".obj", ".ply", ".stl", ".glb", ".gltf"))
    combined = combine_meshes_with_colors(list_of_meshes)

    if is_file:
        glb_path = os.path.splitext(out_path)[0] + ".glb"
        combined.export(glb_path, include_normals=False)
        LOGGER.info("Saved mesh to %s", glb_path)
        return
    # One colored GLB per part (the primary output; mask_palette.json maps mask
    # labels to these mesh_{i:03d}.glb files), colored with the same palette as the
    # combined mesh, plus the combined mesh itself.
    palette = _generate_distinct_colors(len(list_of_meshes))
    for i, mesh in enumerate(list_of_meshes):
        part = mesh.copy()
        col = palette[i % len(palette)]
        part.visual.vertex_colors = np.tile(col[None, :], (len(part.vertices), 1))
        part.export(os.path.join(out_path, f"mesh_{i:03d}.glb"), include_normals=False)
    combined.export(os.path.join(out_path, "mesh_combined.glb"), include_normals=False)
    LOGGER.info("Saved %d part(s) + combined mesh to %s", len(list_of_meshes), out_path)


@torch.no_grad()
def hierarchical_extract_geometry(
    geometric_func,
    device: torch.device,
    bounds=(
        -1.25,
        -1.25,
        -1.25,
        1.25,
        1.25,
        1.25,
    ),
    dense_octree_depth: int = 8,
    hierarchical_octree_depth: int = 9,
    use_coarse_bbox: bool = False,
    coarse_octree_depth: int = 7,
    coarse_band_threshold: float = 1.0,
    coarse_margin_voxels: int = 2,
    isosurface_backend: str = "auto",
):
    # NOTE: default is "auto" (diso/DiffDMC when installed, else skimage). diso's
    # dual marching cubes yields clean watertight parts; the skimage primal
    # (Lewiner) fallback is hardened with allow_degenerate=False below.
    """

    Args:
        geometric_func:
        device:
        bounds:
        dense_octree_depth:
        hierarchical_octree_depth:
        use_coarse_bbox: If True, first estimate a coarse ROI bbox at level `coarse_octree_depth` and shrink `bounds`.
        coarse_octree_depth: Octree depth for the coarse ROI search (e.g., 7).
        coarse_band_threshold: |SDF| threshold used by the band selector for coarse ROI.
        coarse_margin_voxels: Integer voxel margin to expand the coarse ROI on each side.
        isosurface_backend: "skimage" (default), "diso", or "auto". "skimage"
            works out of the box. "diso" uses the optional DiffDMC backend
            (pip install diso). "auto" uses diso when installed and falls back
            to skimage.
    Returns:

    """
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

    bbox_min = torch.tensor(bounds[0:3]).to(device)
    bbox_max = torch.tensor(bounds[3:6]).to(device)
    bbox_size = bbox_max - bbox_min

    # Optional: coarse ROI bbox estimation to shrink bounds
    if use_coarse_bbox and coarse_octree_depth is not None:
        try:
            xyz_samples_c, grid_size_c, _ = generate_dense_grid_points_gpu(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                octree_depth=coarse_octree_depth,
                indexing="ij",
            )
            # [G, G, G]
            grid_logits_c = geometric_func(xyz_samples_c.unsqueeze(0)).view(
                grid_size_c[0], grid_size_c[1], grid_size_c[2]
            )
            # band selection on coarse grid
            coarse_coords = find_candidates_band(
                grid_logits_c, band_threshold=coarse_band_threshold, apply_sigmoid=False
            )
            if coarse_coords.numel() > 0:
                # Clamp with margins
                Gc = int(2**coarse_octree_depth)

                min_idx = coarse_coords.min(dim=0).values
                max_idx = coarse_coords.max(dim=0).values
                min_idx = torch.clamp(min_idx - coarse_margin_voxels, 0, Gc - 1)
                max_idx = torch.clamp(max_idx + coarse_margin_voxels, 0, Gc - 1)

                # Map voxel indices (0..Gc-1) to world coordinates using denom = Gc-1
                # Expand by voxel margins purely in index space (already clamped), then map to world space
                denom = max(Gc - 1, 1)
                new_min = bbox_min + (min_idx.to(bbox_min.dtype) / denom) * bbox_size
                new_max = bbox_min + (max_idx.to(bbox_min.dtype) / denom) * bbox_size

                # Sanity check to avoid degenerate boxes
                if torch.all(new_max > new_min):
                    bbox_min, bbox_max = new_min, new_max
                    bbox_size = bbox_max - bbox_min
        except Exception as _e:
            # Fail-safe: keep original bounds if anything goes wrong
            pass

    xyz_samples, grid_size, length = generate_dense_grid_points_gpu(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_depth=dense_octree_depth,
        indexing="ij",
    )
    # Store the SDF grid in bf16: sigmoid() saturates to exactly +/-1 for |x| > ~5.5
    # in bf16, so find_candidates_band selects a real near-surface band (not every
    # voxel as in fp32), and the bf16 quantization denoises the field (fewer
    # spurious zero-crossings / floaters). Query COORDS stay fp32 (bf16 is too
    # coarse for a 512^3 grid). Matches the TripoSG fp16 behavior.
    grid_logits = (
        geometric_func(xyz_samples.unsqueeze(0))
        .view(grid_size[0], grid_size[1], grid_size[2])
        .to(torch.bfloat16)
    )
    for i in range(hierarchical_octree_depth - dense_octree_depth):
        curr_octree_depth = dense_octree_depth + i + 1
        # upsample
        grid_size = 2**curr_octree_depth
        # interpolate upcasts bf16 -> fp32; keep the grid in bf16 so the scatter
        # (all_logits is bf16) matches and the bf16 quantization holds end-to-end.
        high_res_occupancy = parallel_zoom(grid_logits, 2).to(torch.bfloat16)

        band_threshold = 1.0
        edge_coords = find_candidates_band(grid_logits, band_threshold)
        expanded_coords = expand_edge_region_fast(
            edge_coords, grid_size=int(grid_size / 2)
        )
        # Convert expanded grid indices to world coordinates for queries
        # expanded_coords shape: [N, 3], indices in [0, grid_size-1]
        if expanded_coords.numel() == 0:
            grid_logits = high_res_occupancy
            torch.cuda.empty_cache()
            continue
        denom = max(grid_size - 1, 1)
        expanded_coords_f = expanded_coords.to(torch.float32)
        expanded_coords_world = bbox_min + (expanded_coords_f / denom) * bbox_size

        # Query network only at expanded coords (world space)
        all_logits = (
            geometric_func(expanded_coords_world.unsqueeze(0))[0].view(-1).to(torch.bfloat16)
        )

        # Scatter results back to the high-res occupancy grid using the original indices
        expanded_coords = expanded_coords.type(torch.IntTensor)
        high_res_occupancy[
            expanded_coords[:, 0], expanded_coords[:, 1], expanded_coords[:, 2]
        ] = all_logits

        grid_logits = high_res_occupancy
        torch.cuda.empty_cache()
    grid_logits = grid_logits.float().contiguous()
    backend = (isosurface_backend or "auto").lower()
    try:
        if backend in {"auto", "diso"}:
            try:
                from diso import DiffDMC

                LOGGER.info(
                    "Extracting iso-surface with backend=diso "
                    "(requested=%s, grid_size=%d).",
                    backend,
                    2**hierarchical_octree_depth,
                )
                dmc = DiffDMC(dtype=torch.float32).to(grid_logits.device)
                sdf = (-grid_logits / (2**hierarchical_octree_depth)).contiguous()
                vertices_t, faces_t = dmc(
                    sdf, deform=None, return_quads=False, normalize=False
                )
                vertices = vertices_t.detach().cpu().numpy()
                faces = faces_t.detach().cpu().numpy()
                vertices = (
                    vertices / (2**hierarchical_octree_depth) * bbox_size.cpu().numpy()
                    + bbox_min.cpu().numpy()
                )
                return (vertices.astype(np.float32), np.ascontiguousarray(faces))
            except ImportError as e:
                if backend == "diso":
                    raise ImportError(
                        "isosurface_backend='diso' requires the optional 'diso' "
                        "package. Install it with `pip install diso`, or use "
                        "`--isosurface_backend skimage` (default)."
                    ) from e
                LOGGER.info(
                    "Iso-surface backend auto requested; diso is unavailable, "
                    "falling back to skimage."
                )
                # backend == "auto": fall back to skimage below
            except Exception:
                if backend == "diso":
                    raise

        # allow_degenerate=False so skimage culls the near-zero-area sliver/needle
        # triangles that primal (Lewiner) marching cubes emits where the iso-surface
        # grazes a grid vertex. Keeping them (the old default) produced visibly spiky
        # parts once skimage became the active backend; matches the standalone
        # marching_cubes() helper and Dora's skimage fallback.
        LOGGER.info(
            "Extracting iso-surface with backend=skimage "
            "(requested=%s, grid_size=%d).",
            backend,
            2**hierarchical_octree_depth,
        )
        vertices, faces, normals, _ = sk_marching_cubes(
            grid_logits.cpu().numpy(),
            0,
            method="lewiner",
            gradient_direction="ascent",
            allow_degenerate=False,
        )
        vertices = (
            vertices / (2**hierarchical_octree_depth) * bbox_size.cpu().numpy()
            + bbox_min.cpu().numpy()
        )
        mesh_v_f = (vertices.astype(np.float32), np.ascontiguousarray(faces))
    except Exception:
        torch.cuda.empty_cache()
        mesh_v_f = (None, None)

    return mesh_v_f
