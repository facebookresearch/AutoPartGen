# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

# BriaRMBG (RMBG-1.4) is the background remover TripoSG uses. It is a pure-PyTorch
# model loaded through transformers + trust_remote_code (no onnxruntime).
# NOTE: RMBG-1.4 is released under the non-commercial bria-rmbg-1.4 license;
# commercial use requires a separate agreement with BRIA.
RMBG_MODEL = "briaai/RMBG-1.4"


@lru_cache(maxsize=1)
def _load_rmbg_pipeline():
    """Load (once) the RMBG-1.4 image-segmentation pipeline on GPU when available."""
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on optional installs.
        raise RuntimeError(
            "Background removal is enabled by default but `transformers` is not "
            "installed. Install it (`pip install transformers`) or pass "
            "--no_remove_background."
        ) from exc

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "image-segmentation",
        model=RMBG_MODEL,
        trust_remote_code=True,
        device=device,
    )


def has_valid_alpha(image: Image.Image, min_ratio: float = 0.01) -> bool:
    """Return True when an image already has a usable foreground alpha mask."""
    if image.mode not in ("RGBA", "LA"):
        return False
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    transparent = np.mean(alpha <= 8)
    opaque = np.mean(alpha >= 247)
    return transparent >= min_ratio and opaque >= min_ratio


def remove_background(image: Image.Image) -> Image.Image:
    """Remove the background and return an RGBA image.

    TripoSG runs BriaRMBG before image conditioning; we use that same model
    (RMBG-1.4) here. Images that already carry a valid alpha mask are passed
    through unchanged.
    """
    if has_valid_alpha(image):
        return image.convert("RGBA")

    pipe = _load_rmbg_pipeline()
    rgb = image.convert("RGB")

    # return_mask=True yields a single-channel foreground mask at the input size;
    # composite it onto the original RGB so the colors are preserved exactly.
    mask = pipe(rgb, return_mask=True)
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(np.asarray(mask))
    mask = mask.convert("L")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.BILINEAR)

    rgba = rgb.convert("RGBA")
    rgba.putalpha(mask)
    return rgba
