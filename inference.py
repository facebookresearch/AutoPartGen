# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

from autopartgen.api import GenerationOptions, generate_parts, load_pipeline
from autopartgen.utils.io import get_simple_logger
from autopartgen.utils.mesh import DEFAULT_SIMPLIFY_FACES


def main():
    parser = argparse.ArgumentParser(
        description=(
            "AutoPartGen inference: generate 3D part meshes from image, mesh, "
            "and mask inputs."
        )
    )
    parser.add_argument("--config", default=None, help="Path to inference config YAML.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument("--image", default=None, help="Input object image.")
    parser.add_argument(
        "--mask",
        default=None,
        help="Optional indexed part mask image; requires --image.",
    )
    parser.add_argument(
        "--mesh",
        default=None,
        help="Optional input mesh used as whole-object geometry.",
    )
    parser.add_argument(
        "--output_path",
        "--output_dir",
        dest="output_path",
        default="outputs",
        help="Output directory for mesh_combined.glb and per-part mesh_*.glb files.",
    )
    parser.add_argument(
        "--part_cfg_scale",
        "--image_cfg_scale",
        dest="image_cfg_scale",
        type=float,
        default=None,
        help="Override the selected mode's image guidance scale from the config.",
    )
    parser.add_argument(
        "--geometry_cfg_scale",
        type=float,
        default=None,
        help="Override the selected mode's geometry guidance scale from the config.",
    )
    parser.add_argument(
        "--mask_image_cfg_scale",
        type=float,
        default=None,
        help="Override image guidance for mask modes only.",
    )
    parser.add_argument(
        "--mask_geometry_cfg_scale",
        type=float,
        default=None,
        help="Override geometry guidance for mask modes only.",
    )
    parser.add_argument(
        "--whole_cfg_scale",
        type=float,
        default=None,
        help="Override image-to-whole guidance scale from the config.",
    )
    parser.add_argument(
        "--max_parts",
        type=int,
        default=40,
        help=(
            "Hard cap for autoregressive generation when no mask is provided. "
            "Masked generation follows the number of mask regions."
        ),
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=512,
        help="Iso-surface grid resolution (a power of two); use 256 for faster lower-resolution checks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for diffusion sampling and mesh surface resampling.",
    )
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    parser.add_argument("--iou_grid_size", type=int, default=256)
    parser.add_argument(
        "--duplicate_retries",
        type=int,
        default=3,
        help=(
            "Number of deterministic resampling attempts when a geometry-driven "
            "part is empty or duplicate."
        ),
    )
    parser.add_argument("--part_seed_stride", type=int, default=0)
    parser.add_argument("--retry_seed_stride", type=int, default=1009)
    parser.add_argument(
        "--smooth_iters",
        type=int,
        default=0,
        help="Final Taubin smoothing iterations; default 0 disables smoothing.",
    )
    parser.add_argument(
        "--simplify_faces",
        type=int,
        default=DEFAULT_SIMPLIFY_FACES,
        help=(
            "Per-part face target for quadric simplification; default 50000. "
            "Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--isosurface_backend",
        choices=["skimage", "diso", "auto"],
        default="auto",
        help=(
            "Iso-surface extraction backend. Default 'auto' uses diso (DiffDMC, "
            "recommended for clean watertight parts) when installed and otherwise "
            "falls back to skimage. 'diso' forces DiffDMC (pip install diso). "
            "'skimage' forces the dependency-free marching-cubes fallback."
        ),
    )
    parser.add_argument(
        "--no_mask_visualization",
        action="store_true",
        help="Do not save mask_colored.png and mask_palette.json for indexed-mask inputs.",
    )
    parser.add_argument(
        "--no_remove_background",
        action="store_true",
        help="Disable default image background removal.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable stage logs and diffusion progress bars.",
    )
    parser.set_defaults(use_coarse_bbox=False)
    parser.add_argument(
        "--use_coarse_bbox",
        dest="use_coarse_bbox",
        action="store_true",
        help="Enable coarse ROI cropping during mesh extraction.",
    )
    parser.add_argument(
        "--no_coarse_bbox",
        dest="use_coarse_bbox",
        action="store_false",
        help="Disable coarse ROI cropping during mesh extraction.",
    )
    parser.add_argument("--no_post", "--no_postprocess", action="store_true")
    args = parser.parse_args()

    if args.image is None and args.mesh is None:
        parser.error("Provide --image, --mesh, or both.")
    if args.mask is not None and args.image is None:
        parser.error("--mask requires --image.")

    logger = get_simple_logger("AutoPartGen")
    package_logger = logging.getLogger("autopartgen")
    while package_logger.handlers:
        package_logger.handlers.pop()
    for handler in logger.handlers:
        package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO if not args.no_progress else logging.WARNING)
    package_logger.propagate = False

    pipeline = load_pipeline(
        device=args.device, **({"config": args.config} if args.config else {})
    )
    logger.info("Loaded pipeline on %s", pipeline.device)

    options = GenerationOptions(
        image_cfg_scale=args.image_cfg_scale,
        geometry_cfg_scale=args.geometry_cfg_scale,
        mask_image_cfg_scale=args.mask_image_cfg_scale,
        mask_geometry_cfg_scale=args.mask_geometry_cfg_scale,
        whole_cfg_scale=args.whole_cfg_scale,
        max_parts=args.max_parts,
        grid_size=args.grid_size,
        seed=args.seed,
        postprocess=not args.no_post,
        smooth_iters=args.smooth_iters,
        simplify_faces=args.simplify_faces if args.simplify_faces > 0 else None,
        isosurface_backend=args.isosurface_backend,
        iou_threshold=args.iou_threshold,
        iou_grid_size=args.iou_grid_size,
        duplicate_retries=args.duplicate_retries,
        part_seed_stride=args.part_seed_stride,
        retry_seed_stride=args.retry_seed_stride,
        visualize_mask=not args.no_mask_visualization,
        remove_background=not args.no_remove_background,
        show_progress=not args.no_progress,
        use_coarse_bbox=args.use_coarse_bbox,
    )
    parts = generate_parts(
        pipeline,
        image=args.image,
        mesh=args.mesh,
        mask=args.mask,
        output_dir=args.output_path,
        options=options,
    )
    logger.info("Generated %d part mesh(es); saved to %s", len(parts), args.output_path)


if __name__ == "__main__":
    main()
