# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from autopartgen.utils.mesh import generate_distinct_colors


PathLike = Union[str, os.PathLike]


def _as_index_array(mask: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[-1] >= 3 and np.all(mask[..., 0] == mask[..., 1]) and np.all(
            mask[..., 0] == mask[..., 2]
        ):
            mask = mask[..., 0]
        else:
            raise ValueError("Mask visualization expects an indexed grayscale mask.")
    if mask.ndim != 2:
        raise ValueError("Mask visualization expects a 2D indexed mask.")
    return mask


def ordered_mask_labels(mask: Union[Image.Image, np.ndarray]) -> list[int]:
    """Return non-zero labels in the same order used by masked inference."""
    mask_arr = _as_index_array(mask)
    labels = []
    bboxes = []
    for label in np.unique(mask_arr):
        label_int = int(label)
        if label_int == 0:
            continue
        coords = np.where(mask_arr == label)
        if len(coords[0]) == 0:
            continue
        labels.append(label_int)
        bboxes.append((coords[0].min(), coords[1].min(), coords[0].max()))

    order = sorted(range(len(bboxes)), key=lambda i: (-bboxes[i][2], bboxes[i][1]))
    return [labels[i] for i in order]


def colorize_indexed_mask(mask: Union[Image.Image, np.ndarray]) -> tuple[Image.Image, list[dict]]:
    """Color an indexed mask with the same palette order as exported part meshes."""
    mask_arr = _as_index_array(mask)
    labels = ordered_mask_labels(mask_arr)
    palette = generate_distinct_colors(len(labels))
    rgba = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)

    legend = []
    for part_index, label in enumerate(labels):
        color = palette[part_index].tolist()
        rgba[mask_arr == label] = color
        legend.append(
            {
                "part_index": part_index,
                "mask_label": label,
                "rgba": color,
                "mesh": f"mesh_{part_index:03d}.glb",
            }
        )

    return Image.fromarray(rgba), legend


def _mask_output_paths(output_path: PathLike) -> tuple[Path, Path]:
    output_path = Path(output_path)
    if output_path.suffix.lower() in {".obj", ".ply", ".stl", ".glb", ".gltf"}:
        out_dir = output_path.parent if output_path.parent != Path("") else Path(".")
        stem = output_path.stem
        return out_dir / f"{stem}_mask_colored.png", out_dir / f"{stem}_mask_palette.json"
    return output_path / "mask_colored.png", output_path / "mask_palette.json"


def save_colored_mask(
    mask: Union[Image.Image, np.ndarray], output_path: PathLike
) -> tuple[Path, Path]:
    """Save a colorized indexed mask and its label-to-color legend."""
    colorized, legend = colorize_indexed_mask(mask)
    image_path, legend_path = _mask_output_paths(output_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    colorized.save(image_path)
    legend_path.write_text(json.dumps(legend, indent=2) + "\n", encoding="utf-8")
    return image_path, legend_path
