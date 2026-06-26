# Copyright (c) Meta Platforms, Inc. and affiliates.

"""AutoPartGen inference pipeline.

Pairs the TripoSG diffusion transformer (``PartgenTripoSGDiTModel``) with the
TripoSG VAE (``TripoSGVAEModel``) under v-prediction diffusion. Parts are generated
autoregressively: a "whole" latent is obtained first (generated from the image,
or encoded from a given shape), then parts are sampled one at a time, each
conditioned on the whole plus the history of already-generated parts.

Supported input modes:
  * image -> parts                  (whole generated from the image)
  * mesh -> parts                   (whole encoded from an input mesh)
  * image + mesh -> parts           (image-conditioned parts from an input mesh)
  * image + mask -> parts           (one part per mask region)
  * image + mesh + mask -> parts    (masked parts with image and mesh context)
"""

import logging
import os
import pickle
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import trimesh
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from autopartgen.models.conditioner import ImageConditioner
from autopartgen.models.triposg.autoencoders.autoencoder_kl_triposg import (
    TripoSGVAEModel,
)
from autopartgen.models.partgen_triposg_transformer import PartgenTripoSGDiTModel
from autopartgen.utils.mesh import (
    MAX_N_FACES,
    hierarchical_extract_geometry,
    normalize_trimesh,
    postprocess,
)
from autopartgen.utils.postprocess import postprocess_parts
from autopartgen.utils.sampling import (
    encode_surface,
    IoUCalculator,
    is_near_zero_output,
    resample_surface_from_mesh,
)
from autopartgen.utils import use_seed
from autopartgen.utils.torch_utils import get_torch_dtype

LOGGER = logging.getLogger(__name__)

# Surface sampling (must match the values the checkpoint was trained with).
POINT_CLOUD_SIZE = 32768
FPS_MAX_POINTS = 65536
SURFACE_SAMPLING_TYPE = "fps"
NUM_SURFACE_SAMPLES = 500_000

DEFAULT_GUIDANCE = {
    "whole_cfg_scale": 7.0,
    "image": {"image_cfg_scale": 0.0, "geometry_cfg_scale": 2.0},
    "mesh": {"image_cfg_scale": 0.0, "geometry_cfg_scale": 2.0},
    "image_mesh": {"image_cfg_scale": 0.0, "geometry_cfg_scale": 2.0},
    "image_mask": {"image_cfg_scale": 5.0, "geometry_cfg_scale": 5.0},
    "image_mesh_mask": {"image_cfg_scale": 5.0, "geometry_cfg_scale": 5.0},
}
MIN_INTERNAL_CFG_SCALE = 1.000001

# The TripoSG decoder emits an SDF with the opposite sign convention to the ascent
# marching cubes in hierarchical_extract_geometry, so the decode output is negated
# before iso-surfacing.
EOT_THRESHOLD = 0.1
MAX_PARTS = 40


def _merge_guidance(guidance: Optional[dict]) -> dict:
    merged = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in DEFAULT_GUIDANCE.items()
    }
    if guidance:
        for key, value in guidance.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def _mode_key(image: Optional[Image.Image], mask: Optional[Image.Image], shape) -> str:
    if mask is not None:
        return "image_mesh_mask" if shape is not None else "image_mask"
    if image is not None and shape is not None:
        return "image_mesh"
    if shape is not None:
        return "mesh"
    return "image"


def _resolve_part_guidance(
    defaults: dict,
    image_cfg_scale: Optional[float],
    geometry_cfg_scale: Optional[float],
) -> tuple[float, float, float, List[float]]:
    if image_cfg_scale is None:
        image_cfg_scale = defaults.get("image_cfg_scale", 0.0)
    if geometry_cfg_scale is None:
        geometry_cfg_scale = defaults.get("geometry_cfg_scale", 5.0)

    image_cfg_scale = float(image_cfg_scale)
    geometry_cfg_scale = float(geometry_cfg_scale)
    if image_cfg_scale < 0 or geometry_cfg_scale < 0:
        raise ValueError("CFG scales must be non-negative")

    cfg_scale = max(image_cfg_scale, geometry_cfg_scale, MIN_INTERNAL_CFG_SCALE)
    cond_weights = [image_cfg_scale / cfg_scale, geometry_cfg_scale / cfg_scale]
    return image_cfg_scale, geometry_cfg_scale, cfg_scale, cond_weights


def preprocess_masks(masks: Union[Image.Image, np.ndarray]) -> List[np.ndarray]:
    """Split an indexed mask image into per-part binary masks.

    Different non-zero integer values denote different parts. Parts are sorted
    bottom-to-top, then left-to-right by bounding box (the generation order).
    """
    if isinstance(masks, Image.Image):
        masks = np.array(masks)

    part_masks = []
    part_bboxes = []
    for part_num in np.unique(masks):
        if part_num == 0:
            continue
        coords = np.where(masks == part_num)
        if len(coords[0]) == 0:
            continue
        part_masks.append((masks == part_num).astype(np.uint8) * 255)
        part_bboxes.append((coords[0].min(), coords[1].min(), coords[0].max()))

    order = sorted(
        range(len(part_bboxes)), key=lambda i: (-part_bboxes[i][2], part_bboxes[i][1])
    )
    return [part_masks[i] for i in order]


def _load_module_state_dict(
    module: nn.Module, checkpoint_path: str, strict: bool = True
) -> None:
    """Load the ``model`` sub-dict of a partgen checkpoint into ``module``."""
    LOGGER.info("Loading %s weights from %s", type(module).__name__, checkpoint_path)
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    module.load_state_dict(state_dict, strict=strict)


class AutoPartGenPipeline:
    """Autoregressive part-generation pipeline."""

    def __init__(
        self,
        dit_kwargs: dict,
        vae_kwargs: dict,
        conditioner_kwargs: dict,
        scheduler_kwargs: dict,
        guidance_kwargs: Optional[dict] = None,
        float_precision: str = "bfloat16",
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.precision_dtype = get_torch_dtype(float_precision)
        self.guidance = _merge_guidance(guidance_kwargs)

        dit_kwargs = dict(dit_kwargs)
        vae_kwargs = dict(vae_kwargs)
        dit_pth = dit_kwargs.pop("checkpoint_path")
        vae_pth = vae_kwargs.pop("checkpoint_path")
        self.scale_factor = float(dit_kwargs.get("scale_factor", 1.0))
        self.img_size = conditioner_kwargs.get("img_size", 448)

        # Image conditioner (DINOv2-L/14 with registers).
        self.conditioner = ImageConditioner(**conditioner_kwargs).eval().to(device)

        # Diffusion transformer. It builds its own flow-matching sampler from the
        # scheduler args; weights are partgen-native so a strict load is expected.
        dit_kwargs["args_scheduler"] = dict(scheduler_kwargs)
        self.dit = PartgenTripoSGDiTModel(**dit_kwargs)
        _load_module_state_dict(self.dit, dit_pth, strict=True)
        self.dit = self.dit.eval().to(device)

        # TripoSG VAE.
        self.vae = TripoSGVAEModel(**vae_kwargs)
        _load_module_state_dict(self.vae, vae_pth, strict=True)
        self.vae = self.vae.eval().to(device)

        self.num_latents = self.dit.num_latents
        self.latent_dim = self.dit.latent_dim

    def to(self, device: str) -> "AutoPartGenPipeline":
        self.device = device
        self.conditioner.to(device)
        self.dit.to(device)
        self.vae.to(device)
        return self

    # ---- conditioning ----

    @staticmethod
    def _fg_bbox(alpha: np.ndarray):
        """Foreground bbox (x0,y0,x1,y1) expanded 1.2x and recentered, or None.

        Uses the same foreground framing as training-time image preprocessing.
        """
        fg = alpha > 0.8 * 255
        if not fg.any():
            return None
        ys, xs = np.where(fg)
        xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        half = int(max(xmax - xmin, ymax - ymin) * 1.2) // 2
        return (int(cx - half), int(cy - half), int(cx + half), int(cy + half))

    @staticmethod
    def _composite_white(image: Image.Image) -> np.ndarray:
        """RGBA/LA -> RGB ndarray alpha-composited over white; RGB passes through."""
        if image.mode in ("RGBA", "LA"):
            arr = np.array(image.convert("RGBA"))
            a = arr[:, :, 3:4] / 255.0
            return (arr[:, :, :3].astype(np.float32) * a + (1 - a) * 255.0).astype(np.uint8)
        return np.array(image.convert("RGB"))

    def _prep_image(self, image: Image.Image, bbox=None) -> torch.Tensor:
        """Foreground-crop (1.2x) + composite over white -> (3, H, W) tensor.

        Matches the model's conditioning preprocessing. For RGB inputs
        (no alpha) this is a pass-through resize (no crop), preserving prior
        behavior. ``bbox`` overrides the auto foreground bbox (used so masked-part
        images share the whole object's framing).
        """
        rgb = self._composite_white(image)
        if bbox is None and image.mode in ("RGBA", "LA"):
            bbox = self._fg_bbox(np.array(image.convert("RGBA"))[:, :, 3])
        if bbox is not None:
            h, w = rgb.shape[:2]
            x0, y0, x1, y1 = np.clip(bbox, 0, [w, h, w, h])
            if x1 > x0 and y1 > y0:
                rgb = rgb[y0:y1, x0:x1]
        resized = TF.resize(
            Image.fromarray(rgb), (self.img_size, self.img_size),
            interpolation=InterpolationMode.BICUBIC,
        )
        return TF.to_tensor(resized).to(self.device)

    def _features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """DINOv2 patch features for one image -> (1, T, cross_attention_dim)."""
        return self.conditioner(image_tensor[None])

    def _masked_images(
        self, image: Image.Image, mask: Image.Image
    ) -> List[Image.Image]:
        """Composite each part mask onto the image (white background) -> PIL list.

        The whole image is composited over white (so RGBA backgrounds do not bleed
        through), each mask region keeps its pixels (rest -> white), and every part
        is cropped to the SAME whole-object foreground bbox for consistent framing,
        matching the whole-object crop used for conditioning.
        """
        rgb = self._composite_white(image)
        bbox = (
            self._fg_bbox(np.array(image.convert("RGBA"))[:, :, 3])
            if image.mode in ("RGBA", "LA")
            else None
        )
        out = []
        for part in preprocess_masks(mask):
            keep = np.array(
                Image.fromarray(part).resize(image.size, Image.NEAREST)
            ) > 0
            comp = rgb.copy()
            comp[~keep] = 255
            if bbox is not None:
                h, w = comp.shape[:2]
                x0, y0, x1, y1 = np.clip(bbox, 0, [w, h, w, h])
                if x1 > x0 and y1 > y0:
                    comp = comp[y0:y1, x0:x1]
            out.append(Image.fromarray(comp))
        return out

    # ---- latents ----

    def _zero_history(self) -> torch.Tensor:
        """Start-of-sequence history latent: encode of an empty surface (scaled)."""
        zeros = torch.zeros((1, POINT_CLOUD_SIZE, 6), device=self.device)
        return encode_surface(zeros, self.vae, self.scale_factor)

    def _encode_shape(
        self, mesh: trimesh.Trimesh, sample_posterior: bool = False
    ) -> torch.Tensor:
        """Encode a given mesh's surface to the scaled whole-object latent."""
        mesh = normalize_trimesh(mesh.copy(), scale=1.0, ord=np.inf)
        surface = resample_surface_from_mesh(
            [mesh],
            POINT_CLOUD_SIZE,
            SURFACE_SAMPLING_TYPE,
            FPS_MAX_POINTS,
            self.device,
            num_surface_samples=NUM_SURFACE_SAMPLES,
        )
        return encode_surface(
            surface, self.vae, self.scale_factor, sample_posterior=sample_posterior
        )

    def _history_from_meshes(
        self, meshes: List[trimesh.Trimesh], sample_posterior: bool = False
    ) -> torch.Tensor:
        surface = resample_surface_from_mesh(
            meshes,
            POINT_CLOUD_SIZE,
            SURFACE_SAMPLING_TYPE,
            FPS_MAX_POINTS,
            self.device,
            num_surface_samples=NUM_SURFACE_SAMPLES,
        )
        return encode_surface(
            surface, self.vae, self.scale_factor, sample_posterior=sample_posterior
        )

    def _generate_whole(
        self,
        whole_feat: torch.Tensor,
        drop_feat: torch.Tensor,
        cfg_scale: float,
        seed: int,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Sample a whole-object latent from the image (scaled). Plain CFG.

        The whole is autoregressive element 0. It uses the full-image feature in
        both image slots ([whole; whole]) to match the gen-whole convention.
        """
        img_cond = torch.cat([whole_feat, whole_feat], dim=-1)
        img_uncond = torch.cat([drop_feat, drop_feat], dim=-1)
        sot = torch.zeros(
            (1, 1, self.num_latents, self.latent_dim),
            device=self.device,
            dtype=whole_feat.dtype,
        )
        seeds = torch.arange(seed, seed + 1, device=self.device)
        whole_native = self.dit.sample_with_condition(
            cond=[img_cond, sot],
            batch_seeds=seeds,
            uncond=[img_uncond, torch.zeros_like(sot)],
            cfg_scale=cfg_scale,
            show_progress=show_progress,
            progress_desc="whole diffusion",
        )
        return whole_native * self.scale_factor

    def _sample_part(
        self,
        img_cond: torch.Tensor,
        img_uncond: torch.Tensor,
        geo: torch.Tensor,
        cfg_scale: float,
        cond_weights: List[float],
        seed: int,
        sigma_history: Optional[torch.Tensor] = None,
        sigma_floor: float = 0.0,
        show_progress: bool = False,
        progress_desc: str = "part diffusion",
    ) -> torch.Tensor:
        """Sample one part latent (native space) given image + geometry context."""
        seeds = torch.arange(seed, seed + 1, device=self.device)
        return self.dit.sample_with_condition(
            cond=[[img_cond, geo], [img_uncond, geo]],
            batch_seeds=seeds,
            uncond=[img_uncond, torch.zeros_like(geo)],
            cfg_scale=cfg_scale,
            cond_weights=cond_weights,
            sigma_history=sigma_history,
            sigma_floor=sigma_floor,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

    # ---- decoding ----

    @torch.no_grad()
    def decode_mesh(
        self,
        latent: torch.Tensor,
        grid_size: int,
        mesh_postprocessing: bool = True,
        simplify_faces: Optional[int] = None,
        isosurface_backend: str = "auto",
        use_coarse_bbox: bool = False,
    ) -> trimesh.Trimesh:
        """Decode a part latent (N, C) into a mesh via hierarchical iso-surfacing."""
        assert latent.ndim == 2
        if grid_size <= 0 or (grid_size & (grid_size - 1)) != 0:
            raise ValueError(
                "grid_size must be a positive power of two, e.g. 256 or 512"
            )
        octree_depth = int(np.log2(grid_size))
        dense_depth = int(
            os.environ.get("APG_DENSE_DEPTH", str(max(5, octree_depth - 1)))
        )
        hierarchical_depth = int(os.environ.get("APG_HIER_DEPTH", str(octree_depth)))
        if dense_depth > hierarchical_depth:
            raise ValueError("APG_DENSE_DEPTH must be <= APG_HIER_DEPTH")
        LOGGER.info(
            "Decoding mesh with grid_size=%d, dense_octree_depth=%d, "
            "hierarchical_octree_depth=%d, isosurface_backend=%s.",
            grid_size,
            dense_depth,
            hierarchical_depth,
            isosurface_backend,
        )
        z = latent[None]
        device = latent.device

        def geometric_func(x: torch.Tensor) -> torch.Tensor:
            with torch.autocast("cuda", dtype=self.precision_dtype, enabled=self._amp):
                logits = self.vae.decode(z, x).sample  # (1, P)
            # Negate to match the ascent marching cubes (negate_sdf convention).
            return (-logits).view(1, -1, 1).to(x.dtype)

        verts, faces = hierarchical_extract_geometry(
            geometric_func,
            device,
            bounds=(-1.01, -1.01, -1.01, 1.01, 1.01, 1.01),
            dense_octree_depth=dense_depth,
            hierarchical_octree_depth=hierarchical_depth,
            # Coarse ROI crop is opt-in (off by default); the release path keeps
            # the global decode bounds. Pass use_coarse_bbox=True to enable it.
            use_coarse_bbox=use_coarse_bbox,
            isosurface_backend=isosurface_backend,
        )
        if verts is None:
            return trimesh.Trimesh(vertices=[], faces=[])
        mesh = trimesh.Trimesh(verts.astype(np.float32), faces)
        if mesh_postprocessing:
            # NOTE: smoothing is applied as a final cosmetic pass in _run (not here),
            # so it never feeds the autoregressive history / IoU dedup.
            mesh = postprocess(
                mesh,
                simplification=simplify_faces is not None and simplify_faces > 0,
                simplification_n_faces=int(simplify_faces or MAX_N_FACES),
                device=str(device),
            )
        return mesh

    # ---- main entry ----

    @property
    def _amp(self) -> bool:
        return self.device != "cpu" and torch.cuda.is_available()

    @torch.no_grad()
    def __call__(
        self,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        masks: Optional[Image.Image] = None,
        shape: Optional[trimesh.Trimesh] = None,
        image_cfg_scale: Optional[float] = None,
        geometry_cfg_scale: Optional[float] = None,
        mask_image_cfg_scale: Optional[float] = None,
        mask_geometry_cfg_scale: Optional[float] = None,
        mcubes_grid_size: int = 512,
        mesh_postprocessing: bool = True,
        seed: int = 0,
        max_parts: int = MAX_PARTS,
        iou_threshold: float = 0.3,
        iou_grid_size: int = 256,
        smooth_iters: int = 0,
        simplify_faces: Optional[int] = None,
        isosurface_backend: str = "auto",
        duplicate_retries: int = 3,
        part_seed_stride: int = 0,
        retry_seed_stride: int = 1009,
        whole_cfg_scale: Optional[float] = None,
        infer_sigma_history: float = 0.0,
        sample_posterior: bool = False,
        sigma_floor: float = 0.0,
        show_progress: bool = True,
        use_coarse_bbox: bool = False,
        part_save_dir: Optional[str] = None,
        whole_save_path: Optional[str] = None,
    ) -> List[trimesh.Trimesh]:
        """Generate the parts of a single object.

        Args:
            images: input image (or single-element list); required unless ``shape``.
            masks: optional indexed mask image -> one part per region. It can
                be used with image-only or image+mesh conditioning.
            shape: optional mesh/point cloud used as the whole (shape->parts mode).
            image_cfg_scale: image guidance strength for part sampling. ``None`` uses
                the selected mode's config value.
            geometry_cfg_scale: whole/history geometry guidance strength for part
                sampling. ``None`` uses the selected mode's config value.
            mask_image_cfg_scale: optional image-guidance override for mask modes.
                If unset, ``image_cfg_scale`` is used as a generic override.
            mask_geometry_cfg_scale: optional geometry-guidance override for mask
                modes. If unset, ``geometry_cfg_scale`` is used as a generic override.
            mcubes_grid_size: iso-surface grid resolution.
            mesh_postprocessing: floater removal + simplification on each part.
            seed: sampling seed.
            max_parts: hard cap on the number of autoregressive parts.
            iou_threshold: drop a part whose voxel-IoU with the union of prior parts
                exceeds this (near-duplicate); only active for geometry-driven modes.
            iou_grid_size: voxel grid size for near-duplicate detection.
            smooth_iters: optional final Taubin smoothing iterations after AR
                generation. ``0`` disables smoothing.
            simplify_faces: optional per-part face cap for quadric simplification.
            isosurface_backend: "skimage", "diso", or "auto".
            duplicate_retries: retry budget when a sampled part is empty or duplicate.
            part_seed_stride: deterministic seed increment between part positions.
            retry_seed_stride: deterministic seed increment between retries.
            whole_cfg_scale: CFG scale for the image->whole generation step.
            infer_sigma_history: inference-time noise level for the previous-parts
                history latent. The whole-object slot stays clean.
            sample_posterior: use VAE posterior samples instead of posterior mode when
                encoding the input shape/history latents.
            show_progress: print diffusion progress bars.
            use_coarse_bbox: coarse ROI cropping during mesh extraction; opt-in
                (off by default), pass True to enable it. The release path keeps
                the global decode bounds.
            part_save_dir: optional directory where accepted parts are written as soon
                as they are decoded.
            whole_save_path: optional GLB path for the decoded whole-object latent.

        Returns:
            List of ``trimesh.Trimesh`` part meshes.
        """
        if isinstance(images, (list, tuple)):
            images = images[0] if images else None
        if shape is None and images is None:
            raise ValueError("Either `images` or `shape` must be provided")

        with use_seed(int(seed)):
            with torch.autocast("cuda", dtype=self.precision_dtype, enabled=self._amp):
                return self._run(
                    images,
                    masks,
                    shape,
                    image_cfg_scale,
                    mask_image_cfg_scale,
                    mcubes_grid_size,
                    mesh_postprocessing,
                    seed,
                    max_parts,
                    iou_threshold,
                    iou_grid_size,
                    smooth_iters,
                    simplify_faces,
                    isosurface_backend,
                    duplicate_retries,
                    part_seed_stride,
                    retry_seed_stride,
                    whole_cfg_scale,
                    geometry_cfg_scale,
                    mask_geometry_cfg_scale,
                    infer_sigma_history,
                    sample_posterior,
                    sigma_floor,
                    show_progress,
                    use_coarse_bbox,
                    part_save_dir,
                    whole_save_path,
                )

    def _run(
        self,
        image: Optional[Image.Image],
        mask: Optional[Image.Image],
        shape: Optional[trimesh.Trimesh],
        image_cfg_scale: Optional[float],
        mask_image_cfg_scale: Optional[float],
        grid_size: int,
        mesh_postprocessing: bool,
        seed: int,
        max_parts: int,
        iou_threshold: float,
        iou_grid_size: int,
        smooth_iters: int,
        simplify_faces: Optional[int],
        isosurface_backend: str,
        duplicate_retries: int,
        part_seed_stride: int,
        retry_seed_stride: int,
        whole_cfg_scale: Optional[float],
        geometry_cfg_scale: Optional[float],
        mask_geometry_cfg_scale: Optional[float],
        infer_sigma_history: float,
        sample_posterior: bool,
        sigma_floor: float,
        show_progress: bool,
        use_coarse_bbox: bool,
        part_save_dir: Optional[str],
        whole_save_path: Optional[str],
    ) -> List[trimesh.Trimesh]:
        # Image features (whole + the dropped/uncond dummy).
        if image is not None:
            LOGGER.info("Preparing image conditioning.")
            img_t = self._prep_image(image)
            whole_feat = self._features(img_t)
            drop_feat = self._features(torch.zeros_like(img_t))
        else:
            LOGGER.info("Preparing geometry-only conditioning.")
            # shape->parts: no image, but the geometry-driven CFG path still needs a
            # real image embedding slot. The default image guidance is zero, so the
            # content is immaterial; fill it with a zeroed-image embedding.
            zero_img = torch.zeros(
                (3, self.img_size, self.img_size), device=self.device
            )
            whole_feat = drop_feat = self._features(zero_img)
        dummy = torch.cat([drop_feat, drop_feat], dim=-1)
        mode_key = _mode_key(image, mask, shape)
        guidance_defaults = self.guidance.get(mode_key, {})
        whole_scale = (
            float(whole_cfg_scale)
            if whole_cfg_scale is not None
            else float(self.guidance.get("whole_cfg_scale", 5.0))
        )

        # Whole-object latent (scaled context).
        if shape is not None:
            LOGGER.info("Encoding input mesh as whole-object latent.")
            whole_scaled = self._encode_shape(shape, sample_posterior=sample_posterior)
        else:
            LOGGER.info("Generating whole-object latent.")
            whole_scaled = self._generate_whole(
                whole_feat,
                drop_feat,
                whole_scale,
                seed,
                show_progress=show_progress,
            )

        if whole_save_path:
            LOGGER.info("Decoding whole-object latent to %s.", whole_save_path)
            os.makedirs(os.path.dirname(whole_save_path), exist_ok=True)
            whole_mesh = self.decode_mesh(
                whole_scaled[0] / self.scale_factor,
                grid_size,
                mesh_postprocessing=mesh_postprocessing,
                simplify_faces=simplify_faces,
                isosurface_backend=isosurface_backend,
                use_coarse_bbox=use_coarse_bbox,
            )
            if len(whole_mesh.faces) > 0:
                whole_mesh.export(whole_save_path)

        # Mode-specific per-part image conditioning + guidance + iteration count.
        masked_feats = None
        if mask is not None and image is not None:
            LOGGER.info("Preparing masked part conditioning.")
            masked_feats = [
                self._features(self._prep_image(mi))
                for mi in self._masked_images(image, mask)
            ]
            num_iters = len(masked_feats)
            use_iou = False  # one part per mask; no dedup
        else:
            num_iters = max_parts
            use_iou = iou_threshold >= 0

        image_override = image_cfg_scale
        geometry_override = geometry_cfg_scale
        if masked_feats is not None:
            if mask_image_cfg_scale is not None:
                image_override = mask_image_cfg_scale
            if mask_geometry_cfg_scale is not None:
                geometry_override = mask_geometry_cfg_scale

        image_scale, geometry_scale, cfg_scale, cond_weights = _resolve_part_guidance(
            guidance_defaults, image_override, geometry_override
        )
        LOGGER.info(
            "Part guidance for %s: image_cfg_scale=%.3g, geometry_cfg_scale=%.3g.",
            mode_key,
            image_scale,
            geometry_scale,
        )

        if masked_feats is not None:
            LOGGER.info("Generating %d mask-specified part(s).", num_iters)
        else:
            LOGGER.info("Generating up to %d part(s).", num_iters)

        iou_calc = IoUCalculator(grid_size=iou_grid_size) if use_iou else None
        history = self._zero_history()
        parts: List[trimesh.Trimesh] = []
        if part_save_dir:
            os.makedirs(part_save_dir, exist_ok=True)

        for i in range(num_iters):
            if masked_feats is not None:
                part_label = f"{i + 1}/{num_iters}"
                progress_desc = f"part {part_label} diffusion"
            else:
                part_label = f"{i + 1}"
                progress_desc = f"part {part_label} diffusion"

            LOGGER.info("Sampling part %s.", part_label)
            if masked_feats is not None:
                # Match training and original PartGen sampling: image context is
                # [whole-object image ; masked-part image], regardless of whether
                # the whole latent came from an input mesh or image generation.
                # The shape (when given) is carried via `geo`, not these image slots.
                img_cond = torch.cat([whole_feat, masked_feats[i]], dim=-1)
            else:
                img_cond = torch.cat([whole_feat, drop_feat], dim=-1)

            geo = torch.stack([whole_scaled, history], dim=1)  # (1, 2, N, C)
            sigma_history = None
            if infer_sigma_history > 0 and parts:
                sigma = torch.zeros(
                    (geo.shape[0], geo.shape[1]), device=geo.device, dtype=geo.dtype
                )
                sigma[:, 1] = float(infer_sigma_history)
                sigma_b = sigma.view(geo.shape[0], geo.shape[1], 1, 1)
                geo = (1.0 - sigma_b) * geo + sigma_b * torch.randn_like(geo)
                sigma_history = sigma
            max_attempts = 1 if not use_iou else max(1, duplicate_retries + 1)
            accepted = False
            for retry_idx in range(max_attempts):
                part_seed = seed + i * part_seed_stride + retry_idx * retry_seed_stride
                sampled = self._sample_part(
                    img_cond,
                    dummy,
                    geo,
                    cfg_scale,
                    cond_weights,
                    part_seed,
                    sigma_history=sigma_history,
                    sigma_floor=sigma_floor,
                    show_progress=show_progress,
                    progress_desc=progress_desc,
                )

                if use_iou and is_near_zero_output(sampled[0], EOT_THRESHOLD):
                    LOGGER.info(
                        "Reached end-of-sequence before accepting part %d; stopping.",
                        i + 1,
                    )
                    return self._finalize_parts(
                        parts, mesh_postprocessing, smooth_iters, use_iou
                    )

                LOGGER.info("Decoding part %s.", part_label)
                mesh = self.decode_mesh(
                    sampled[0],
                    grid_size,
                    mesh_postprocessing=False,
                    simplify_faces=simplify_faces,
                    isosurface_backend=isosurface_backend,
                    use_coarse_bbox=use_coarse_bbox,
                )
                if len(mesh.faces) == 0:
                    LOGGER.info("Part %d retry %d decoded empty mesh.", i, retry_idx)
                    continue

                if iou_calc is not None and parts:
                    iou = iou_calc.compute_iou(trimesh.util.concatenate(parts), mesh)
                    if iou > iou_threshold:
                        LOGGER.info(
                            "Part %d retry %d IoU %.3f > %.3f; resampling.",
                            i,
                            retry_idx,
                            iou,
                            iou_threshold,
                        )
                        continue

                if mesh_postprocessing:
                    LOGGER.info("Post-processing part %s.", part_label)
                    mesh = postprocess(
                        mesh,
                        simplification=simplify_faces is not None
                        and simplify_faces > 0,
                        simplification_n_faces=int(simplify_faces or MAX_N_FACES),
                        device=str(self.device),
                    )
                    if len(mesh.faces) == 0:
                        LOGGER.info(
                            "Part %d retry %d became empty after post-processing.",
                            i,
                            retry_idx,
                        )
                        continue

                parts.append(mesh)
                if part_save_dir:
                    mesh.export(
                        os.path.join(part_save_dir, f"mesh_{len(parts) - 1:03d}.glb"),
                        include_normals=False,
                    )
                LOGGER.info("Encoding generated-part history after part %d.", i + 1)
                history = self._history_from_meshes(
                    parts, sample_posterior=sample_posterior
                )
                accepted = True
                break

            if use_iou and not accepted:
                LOGGER.info(
                    "Part %d was not accepted after %d attempts; stopping.",
                    i,
                    max_attempts,
                )
                break

        LOGGER.info("Finalizing %d generated part(s).", len(parts))
        return self._finalize_parts(parts, mesh_postprocessing, smooth_iters, use_iou)

    @staticmethod
    def _finalize_parts(
        parts: List[trimesh.Trimesh],
        mesh_postprocessing: bool,
        smoothing_iters: int,
        use_iou: bool,
    ) -> List[trimesh.Trimesh]:
        if mesh_postprocessing and use_iou and len(parts) > 1:
            parts = postprocess_parts(
                parts, scene_handling=False, preserve_groups=True
            )

        parts = [p for p in parts if p is not None and len(p.faces) > 0]

        if mesh_postprocessing and smoothing_iters > 0:
            for p in parts:
                if len(p.faces) > 20:
                    try:
                        trimesh.smoothing.filter_taubin(p, iterations=smoothing_iters)
                    except Exception:
                        pass

        return parts
