# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Small public API for AutoPartGen inference.

This module keeps the release-facing API compact while leaving the lower-level
``AutoPartGenPipeline`` available for advanced use.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional, Union

import trimesh
from PIL import Image

from autopartgen.autopartgen import AutoPartGenPipeline
from autopartgen.utils.background import remove_background as remove_image_background
from autopartgen.utils.io import load_config, load_image
from autopartgen.utils.mask import save_colored_mask
from autopartgen.utils.mesh import DEFAULT_SIMPLIFY_FACES, save_colored_mesh
from autopartgen.utils.torch_utils import select_device


PathLike = Union[str, Path]
ImageLike = Union[PathLike, Image.Image]
MeshLike = Union[PathLike, trimesh.Trimesh]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = resources.files("autopartgen.configs") / "default.yaml"


@dataclass
class GenerationOptions:
    """Inference and mesh export options.

    Defaults target final-quality output. Use ``grid_size=256`` for faster
    lower-resolution checks and ``grid_size=512`` for release-quality meshes.
    """

    image_cfg_scale: Optional[float] = None
    geometry_cfg_scale: Optional[float] = None
    mask_image_cfg_scale: Optional[float] = None
    mask_geometry_cfg_scale: Optional[float] = None
    whole_cfg_scale: Optional[float] = None
    max_parts: int = 40
    grid_size: int = 512
    seed: int = 0
    postprocess: bool = True
    smooth_iters: int = 0
    simplify_faces: Optional[int] = DEFAULT_SIMPLIFY_FACES
    isosurface_backend: str = "auto"
    iou_threshold: float = 0.9
    iou_grid_size: int = 512
    duplicate_retries: int = 3
    part_seed_stride: int = 0
    retry_seed_stride: int = 1009
    infer_sigma_history: float = 0.0
    sample_posterior: bool = False
    sigma_floor: float = 0.0
    visualize_mask: bool = True
    remove_background: bool = True
    show_progress: bool = True
    # Coarse ROI crop is opt-in; the release path keeps the global decode bounds.
    use_coarse_bbox: bool = False


def load_pipeline(
    config: PathLike = DEFAULT_CONFIG,
    device: Optional[str] = None,
) -> AutoPartGenPipeline:
    """Load a configured AutoPartGen pipeline."""

    cfg = load_config(str(config))
    # Anchor relative checkpoint paths to the repo root so the API works from any
    # working directory (the default config uses "checkpoints/..."). Absolute paths
    # and paths that already resolve against the cwd are left untouched.
    for sub in ("dit", "vae"):
        ckpt = cfg.get(sub, {}).get("checkpoint_path")
        if ckpt and not Path(ckpt).is_absolute() and not Path(ckpt).exists():
            cfg[sub]["checkpoint_path"] = str(REPO_ROOT / ckpt)
    return AutoPartGenPipeline(
        dit_kwargs=cfg["dit"],
        vae_kwargs=cfg["vae"],
        conditioner_kwargs=cfg["conditioner"],
        scheduler_kwargs=cfg["scheduler"],
        guidance_kwargs=cfg.get("guidance"),
        float_precision=cfg.get("float_precision", "bfloat16"),
        device=device or select_device(),
    )


def _read_image(
    image: Optional[ImageLike], *, remove_background: bool = False
) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        image_obj = image
    else:
        image_obj = load_image(str(image))
    if remove_background:
        image_obj = remove_image_background(image_obj)
    return image_obj


def _read_mesh(mesh: Optional[MeshLike]) -> Optional[trimesh.Trimesh]:
    if mesh is None:
        return None
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    loaded = trimesh.load(str(mesh), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.dump()))
    return loaded


def _prepare_output_dir(output_dir: Optional[PathLike]) -> Optional[Path]:
    if output_dir is None:
        return None
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Remove only AutoPartGen-generated artifacts from previous runs.
    for stale in path.glob("mesh_*.glb"):
        stale.unlink(missing_ok=True)
    for name in (
        "mesh_combined.glb",
        "mask_colored.png",
        "mask_palette.json",
        "whole_generated.glb",
    ):
        (path / name).unlink(missing_ok=True)

    return path


def generate_parts(
    pipeline: AutoPartGenPipeline,
    *,
    image: Optional[ImageLike] = None,
    mesh: Optional[MeshLike] = None,
    mask: Optional[ImageLike] = None,
    output_dir: Optional[PathLike] = None,
    options: Optional[GenerationOptions] = None,
    whole_save_path: Optional[PathLike] = None,
) -> list[trimesh.Trimesh]:
    """Generate part meshes from image, mesh, image+mesh, or masked input.

    ``mesh`` supplies the whole-object geometry when provided. ``image`` supplies
    image conditioning. ``mask`` is an indexed part mask; each non-zero label is
    decoded as one output part and requires ``image``. Passing ``image``,
    ``mesh``, and ``mask`` together runs image+mesh+mask conditioning.
    """

    if image is None and mesh is None:
        raise ValueError("Provide at least one of `image` or `mesh`.")
    if mask is not None and image is None:
        raise ValueError("`mask` requires `image`.")
    options = options or GenerationOptions()
    part_output_dir = _prepare_output_dir(output_dir)
    image_obj = _read_image(image, remove_background=options.remove_background)
    mask_obj = _read_image(mask, remove_background=False)
    mesh_obj = _read_mesh(mesh)

    parts = pipeline(
        images=image_obj,
        masks=mask_obj,
        shape=mesh_obj,
        image_cfg_scale=options.image_cfg_scale,
        geometry_cfg_scale=options.geometry_cfg_scale,
        mask_image_cfg_scale=options.mask_image_cfg_scale,
        mask_geometry_cfg_scale=options.mask_geometry_cfg_scale,
        whole_cfg_scale=options.whole_cfg_scale,
        mcubes_grid_size=options.grid_size,
        mesh_postprocessing=options.postprocess,
        seed=options.seed,
        max_parts=options.max_parts,
        iou_threshold=options.iou_threshold,
        iou_grid_size=options.iou_grid_size,
        smooth_iters=options.smooth_iters,
        simplify_faces=options.simplify_faces,
        isosurface_backend=options.isosurface_backend,
        duplicate_retries=options.duplicate_retries,
        part_seed_stride=options.part_seed_stride,
        retry_seed_stride=options.retry_seed_stride,
        infer_sigma_history=options.infer_sigma_history,
        sample_posterior=options.sample_posterior,
        sigma_floor=options.sigma_floor,
        show_progress=options.show_progress,
        use_coarse_bbox=options.use_coarse_bbox,
        part_save_dir=str(part_output_dir) if part_output_dir is not None else None,
        whole_save_path=str(whole_save_path) if whole_save_path is not None else None,
    )

    if output_dir is not None:
        save_colored_mesh(parts, str(output_dir))
        if mask_obj is not None and options.visualize_mask:
            save_colored_mask(mask_obj, output_dir)
    return parts


def generate_from_image(
    pipeline: AutoPartGenPipeline,
    image: ImageLike,
    *,
    output_dir: Optional[PathLike] = None,
    options: Optional[GenerationOptions] = None,
) -> list[trimesh.Trimesh]:
    return generate_parts(pipeline, image=image, output_dir=output_dir, options=options)


def generate_from_mesh(
    pipeline: AutoPartGenPipeline,
    mesh: MeshLike,
    *,
    output_dir: Optional[PathLike] = None,
    options: Optional[GenerationOptions] = None,
) -> list[trimesh.Trimesh]:
    return generate_parts(pipeline, mesh=mesh, output_dir=output_dir, options=options)


def generate_from_image_and_mesh(
    pipeline: AutoPartGenPipeline,
    image: ImageLike,
    mesh: MeshLike,
    *,
    output_dir: Optional[PathLike] = None,
    options: Optional[GenerationOptions] = None,
) -> list[trimesh.Trimesh]:
    return generate_parts(
        pipeline, image=image, mesh=mesh, output_dir=output_dir, options=options
    )


def generate_from_image_and_mask(
    pipeline: AutoPartGenPipeline,
    image: ImageLike,
    mask: ImageLike,
    *,
    mesh: Optional[MeshLike] = None,
    output_dir: Optional[PathLike] = None,
    options: Optional[GenerationOptions] = None,
) -> list[trimesh.Trimesh]:
    return generate_parts(
        pipeline,
        image=image,
        mask=mask,
        mesh=mesh,
        output_dir=output_dir,
        options=options,
    )
