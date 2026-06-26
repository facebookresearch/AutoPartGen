# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe
import inspect
from functools import wraps
from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy as np

import torch

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]


def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)


def sphere_hammersley_sequence(n, num_samples, offset=(0, 0), remap=False):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    if remap:
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def get_device(args, kwargs):
    device = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            if device is None:
                device = arg.device
            elif device != arg.device:
                raise ValueError("All tensors must be on the same device.")
    return device


def get_args_order(func, args, kwargs):
    """
    Get the order of the arguments of a function.
    """
    names = inspect.getfullargspec(func).args
    names_idx = {name: i for i, name in enumerate(names)}
    args_order = []
    kwargs_order = {}
    for name, arg in kwargs.items():
        if name in names:
            kwargs_order[name] = names_idx[name]
            names.remove(name)
    for i, arg in enumerate(args):
        if i < len(names):
            args_order.append(names_idx[names[i]])
    return args_order, kwargs_order


def suppress_traceback(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            e.__traceback__ = e.__traceback__.tb_next.tb_next
            raise

    return wrapper


def broadcast_args(args, kwargs, args_dim, kwargs_dim):
    spatial = []
    for arg, arg_dim in zip(
        args + list(kwargs.values()), args_dim + list(kwargs_dim.values())
    ):
        if isinstance(arg, torch.Tensor) and arg_dim is not None:
            arg_spatial = arg.shape[: arg.ndim - arg_dim]
            if len(arg_spatial) > len(spatial):
                spatial = [1] * (len(arg_spatial) - len(spatial)) + spatial
            for j in range(len(arg_spatial)):
                if spatial[-j] < arg_spatial[-j]:
                    if spatial[-j] == 1:
                        spatial[-j] = arg_spatial[-j]
                    else:
                        raise ValueError("Cannot broadcast arguments.")
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and args_dim[i] is not None:
            args[i] = torch.broadcast_to(
                arg, [*spatial, *arg.shape[arg.ndim - args_dim[i] :]]
            )
    for key, arg in kwargs.items():
        if isinstance(arg, torch.Tensor) and kwargs_dim[key] is not None:
            kwargs[key] = torch.broadcast_to(
                arg, [*spatial, *arg.shape[arg.ndim - kwargs_dim[key] :]]
            )
    return args, kwargs, spatial


@suppress_traceback
def batched(*dims):
    """
    Decorator that allows a function to be called with batched arguments.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, device=torch.device("cpu"), **kwargs):
            args = list(args)
            # get arguments dimensions
            args_order, kwargs_order = get_args_order(func, args, kwargs)
            args_dim = [dims[i] for i in args_order]
            kwargs_dim = {key: dims[i] for key, i in kwargs_order.items()}
            # convert to torch tensor
            device = get_device(args, kwargs) or device
            for i, arg in enumerate(args):
                if isinstance(arg, (Number, list, tuple)) and args_dim[i] is not None:
                    args[i] = torch.tensor(arg, device=device)
            for key, arg in kwargs.items():
                if (
                    isinstance(arg, (Number, list, tuple))
                    and kwargs_dim[key] is not None
                ):
                    kwargs[key] = torch.tensor(arg, device=device)
            # broadcast arguments
            args, kwargs, spatial = broadcast_args(args, kwargs, args_dim, kwargs_dim)
            for i, (arg, arg_dim) in enumerate(zip(args, args_dim)):
                if isinstance(arg, torch.Tensor) and arg_dim is not None:
                    args[i] = arg.reshape([-1, *arg.shape[arg.ndim - arg_dim :]])
            for key, arg in kwargs.items():
                if isinstance(arg, torch.Tensor) and kwargs_dim[key] is not None:
                    kwargs[key] = arg.reshape(
                        [-1, *arg.shape[arg.ndim - kwargs_dim[key] :]]
                    )
            # call function
            results = func(*args, **kwargs)
            type_results = type(results)
            results = list(results) if isinstance(results, (tuple, list)) else [results]
            # restore spatial dimensions
            for i, result in enumerate(results):
                results[i] = result.reshape([*spatial, *result.shape[1:]])
            if type_results == tuple:
                results = tuple(results)
            elif type_results == list:
                results = list(results)
            else:
                results = results[0]
            return results

        return wrapper

    return decorator


def _group(
    values: torch.Tensor,
    required_group_size: Optional[int] = None,
    return_values: bool = False,
) -> Tuple[Union[List[torch.Tensor], torch.Tensor, Tuple], Optional[torch.Tensor]]:
    """
    Group values into groups with identical values.

    Args:
        values (torch.Tensor): [N] values to group
        required_group_size (int, optional): required group size. Defaults to None.
        return_values (bool, optional): return values of groups. Defaults to False.

    Returns:
        group (Union[List[torch.Tensor], torch.Tensor]): list of groups or group indices. It will be a list of groups if required_group_size is None, otherwise a tensor of group indices.
        group_values (Optional[torch.Tensor]): values of groups. Only returned if return_values is True.
    """
    sorted_values, indices = torch.sort(values)
    nondupe = torch.cat(
        [
            torch.tensor([True], dtype=torch.bool, device=values.device),
            sorted_values[1:] != sorted_values[:-1],
        ]
    )
    nondupe_indices = torch.cumsum(nondupe, dim=0) - 1
    counts = torch.bincount(nondupe_indices)
    if required_group_size is None:
        groups = torch.split(indices, counts.tolist())
        if return_values:
            group_values = sorted_values[nondupe]
            return groups, group_values
        else:
            return groups
    else:
        counts = counts[nondupe_indices]
        groups = indices[counts == required_group_size].reshape(-1, required_group_size)
        if return_values:
            group_values = sorted_values[nondupe][
                counts[nondupe] == required_group_size
            ]
            return groups, group_values
        else:
            return groups


@batched(1, 1, 1)
def view_look_at(
    eye: torch.Tensor, look_at: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    """
    Get OpenGL view matrix looking at something

    Args:
        eye (torch.Tensor): [..., 3] the eye position
        look_at (torch.Tensor): [..., 3] the position to look at
        up (torch.Tensor): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (torch.Tensor): [..., 4, 4], view matrix
    """
    N = eye.shape[0]
    z = eye - look_at
    x = torch.cross(up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    # x = torch.cross(y, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    R = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(R, eye[..., None])
    ret = torch.zeros((N, 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = R
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.0
    return ret


def compute_edges(
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute edges of a mesh.

    Args:
        faces (torch.Tensor): [T, 3] triangular face indices

    Returns:
        edges (torch.Tensor): [E, 2] edge indices
        face2edge (torch.Tensor): [T, 3] mapping from face to edge
        counts (torch.Tensor): [E] degree of each edge
    """
    T = faces.shape[0]
    edges = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
    )  # [3T, 2]
    edges = torch.sort(edges, dim=1).values
    edges, inv_map, counts = torch.unique(
        edges, return_inverse=True, return_counts=True, dim=0
    )
    face2edge = inv_map.view(3, T).T
    return edges, face2edge, counts


def compute_connected_components(
    faces: torch.Tensor,
    edges: Optional[torch.Tensor] = None,
    face2edge: Optional[torch.Tensor] = None,
) -> Union[List, Tuple]:
    """
    Compute connected faces of a mesh.

    Args:
        faces (torch.Tensor): [T, 3] triangular face indices
        edges (torch.Tensor, optional): [E, 2] edge indices. Defaults to None.
        face2edge (torch.Tensor, optional): [T, 3] mapping from face to edge. Defaults to None.
            NOTE: If edges and face2edge are not provided, they will be computed.

    Returns:
        components (List[torch.Tensor]): list of connected faces
    """
    T = faces.shape[0]
    if edges is None or face2edge is None:
        edges, face2edge, _ = compute_edges(faces)
    E = edges.shape[0]

    labels = torch.arange(T, dtype=torch.int32, device=faces.device)
    while True:
        edge_labels = torch.scatter_reduce(
            torch.zeros(E, dtype=torch.int32, device=faces.device),
            0,
            face2edge.flatten().long(),
            labels.view(-1, 1).expand(-1, 3).flatten(),
            reduce="amin",
            include_self=False,
        )
        new_labels = torch.min(edge_labels[face2edge], dim=-1).values
        if torch.equal(labels, new_labels):
            break
        labels = new_labels

    components = _group(labels)

    return components


def compute_edge_connected_components(
    edges: torch.Tensor,
) -> Union[List, Tuple]:
    """
    Compute connected edges of a mesh.

    Args:
        edges (torch.Tensor): [E, 2] edge indices

    Returns:
        components (List[torch.Tensor]): list of connected edges
    """
    E = edges.shape[0]

    # Re-index edges
    verts, edges = torch.unique(edges.flatten(), return_inverse=True)
    edges = edges.view(-1, 2)
    V = verts.shape[0]

    labels = torch.arange(E, dtype=torch.int32, device=edges.device)
    while True:
        vertex_labels = torch.scatter_reduce(
            torch.zeros(V, dtype=torch.int32, device=edges.device),
            0,
            edges.flatten().long(),
            labels.view(-1, 1).expand(-1, 2).flatten(),
            reduce="amin",
            include_self=False,
        )
        new_labels = torch.min(vertex_labels[edges], dim=-1).values
        if torch.equal(labels, new_labels):
            break
        labels = new_labels

    components = _group(labels)

    return components


def compute_dual_graph(
    face2edge: torch.Tensor,
) -> Tuple[int, Optional[torch.Tensor]]:
    """
    Compute dual graph of a mesh.

    Args:
        face2edge (torch.Tensor): [T, 3] mapping from face to edge.

    Returns:
        dual_edges (torch.Tensor): [DE, 2] face indices of dual edges
        dual_edge2edge (torch.Tensor): [DE] mapping from dual edge to edge
    """
    all_edge_indices = face2edge.flatten()  # [3T]
    dual_edges, dual_edge2edge = _group(
        all_edge_indices, required_group_size=2, return_values=True
    )
    assert isinstance(dual_edges, torch.Tensor)
    dual_edges = dual_edges // face2edge.shape[1]
    return dual_edges, dual_edge2edge


def remove_unreferenced_vertices(
    faces: torch.Tensor, *vertice_attrs, return_indices: bool = False
) -> Tuple[torch.Tensor, ...]:
    """
    Remove unreferenced vertices of a mesh.
    Unreferenced vertices are removed, and the face indices are updated accordingly.

    Args:
        faces (torch.Tensor): [T, P] face indices
        *vertice_attrs: vertex attributes

    Returns:
        faces (torch.Tensor): [T, P] face indices
        *vertice_attrs: vertex attributes
        indices (torch.Tensor, optional): [N] indices of vertices that are kept. Defaults to None.
    """
    P = faces.shape[-1]
    fewer_indices, inv_map = torch.unique(faces, return_inverse=True)
    faces = inv_map.to(torch.int32).reshape(-1, P)
    ret = [faces]
    for attr in vertice_attrs:
        ret.append(attr[fewer_indices])
    if return_indices:
        ret.append(fewer_indices)
    return tuple(ret)
