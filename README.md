<div align="center">
<h1>AutoPartGen: Autoregressive 3D Part Generation and Discovery</h1>

<p>
<a href="https://arxiv.org/abs/2507.13346"><img src="https://img.shields.io/badge/arXiv-2507.13346-b31b1b" alt="arXiv"></a>
<a href="https://silent-chen.github.io/AutoPartGen/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/facebook/autopartgen"><img src="https://img.shields.io/badge/Model-facebook%2Fautopartgen-yellow" alt="Model"></a>
</p>

<p>
<b><a href="https://www.robots.ox.ac.uk/~vgg/">Visual Geometry Group, University of Oxford</a></b>;
<b><a href="https://ai.facebook.com/research/">Meta AI</a></b>
</p>

<p>
<a href="https://silent-chen.github.io/">Minghao Chen</a>,
<a href="https://jytime.github.io/">Jianyuan Wang</a>,
<a href="https://www.shapovalov.ro/">Roman Shapovalov</a>,
<a href="https://www.tmonnier.com/">Tom Monnier</a>,
<a href="https://hyblue.github.io/">Hyunyoung Jung</a>,
<a href="https://wdilin.github.io/">Dilin Wang</a>,
<a href="https://scholar.google.com/citations?user=8KF99lYAAAAJ&hl=en">Rakesh Ranjan</a>,
<a href="https://scholar.google.de/citations?user=n9nXAPcAAAAJ&hl=en">Iro Laina</a>,
<a href="https://www.robots.ox.ac.uk/~vedaldi/">Andrea Vedaldi</a>
</p>
</div>

AutoPartGen generates compositional 3D objects in an autoregressive manner. It
can produce a set of part meshes from an object image, an indexed part mask, an
object mesh, or combinations of these inputs.

> [!IMPORTANT]
> This repository provides a reimplementation of AutoPartGen based on TripoSG
> components and released checkpoints, since the original internal model is
> subject to release constraints. Its results are expected to differ from, and
> may underperform, the original system reported in the paper.

## Pretrained Models

Download the released weights from
[facebook/autopartgen](https://huggingface.co/facebook/autopartgen) and place
them in `checkpoints/`:

| Component | Expected path |
| :--- | :--- |
| Part-generation DiT, default 2048-latent release | `checkpoints/autopartgen_dit.pth` |
| Shape VAE | `checkpoints/autopartgen_vae.pth` |

```bash
hf download facebook/autopartgen \
  autopartgen_dit.pth autopartgen_vae.pth \
  --local-dir checkpoints
```

We also provide an optional 4096-latent-token DiT checkpoint, finetuned from the
2048-token model for another 50k steps, which may improve fine-detail and part
modeling for some inputs at a higher memory and runtime cost:

```bash
hf download facebook/autopartgen \
  autopartgen_dit_4096.pth autopartgen_vae.pth \
  --local-dir checkpoints
```

If the model repository requires authentication, run `hf auth login` first.

## Quick Start

The recommended setup creates the conda environment first, installs PyTorch and
torchvision explicitly, then installs the remaining AutoPartGen dependencies.
This keeps PyTorch importable before optional CUDA extensions such as `diso` are
built.

```bash
conda create -n autopartgen python=3.10 pip -y
conda activate autopartgen
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

For CPU-only development, use the same order but install the CPU PyTorch wheels:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

The release runtime works without optional CUDA extensions. The default
iso-surface backend is `auto`: it uses DiffDMC through the optional `diso`
package when installed, and otherwise falls back to scikit-image marching
cubes.

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
PY
pip install diso --no-build-isolation
```

`diso` imports PyTorch during its build and compiles CUDA kernels, so install it
only after `torch` is importable. It also needs a CUDA toolkit matching your
PyTorch build (`nvcc`, `cuda_runtime.h`, and the thrust/cccl headers) on the
build path. For the pinned cu121 PyTorch build, install CUDA 12.1 toolkit
packages first if `nvcc` is not already available:

```bash
conda install -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev cuda-cccl
pip install diso --no-build-isolation
```

Pass `--isosurface_backend skimage` to force the dependency-free fallback, or
`diso` to require DiffDMC.

Image background removal is enabled by default and uses BriaRMBG (RMBG-1.4) via
`transformers`. The first run downloads the model. RMBG-1.4 is released under a
non-commercial license. Pass `--no_remove_background` if the input already has
the desired background or alpha mask.

Optional scene post-processing backends can be installed with:

```bash
pip install ".[postprocess]"
```

## Inference

Image-conditioned generation:

```bash
python inference.py \
  --image examples/image/apple_character/image.png \
  --output_path outputs/apple_character
```

4096-latent-token image-conditioned generation:

```bash
python inference.py \
  --config autopartgen/configs/default_4096.yaml \
  --image examples/image/apple_character/image.png \
  --output_path outputs/apple_character_4096
```

Mesh-conditioned generation:

```bash
python inference.py \
  --mesh examples/mesh/potted_flower/mesh.glb \
  --output_path outputs/potted_flower_mesh
```

The `--mesh` input can be any whole-object mesh that `trimesh` can load,
including scanned, reconstructed, or third-party generated meshes.

Image, mesh, and indexed-mask conditioning:

```bash
python inference.py \
  --image examples/image_mesh_mask/robot/image.png \
  --mesh examples/image_mesh_mask/robot/mesh.glb \
  --mask examples/image_mesh_mask/robot/mask.png \
  --output_path outputs/robot_image_mesh_mask
```

The output directory contains one GLB per accepted part (`mesh_000.glb`,
`mesh_001.glb`, ...) plus the combined part mesh as `mesh_combined.glb`. For indexed-mask inputs,
AutoPartGen saves `mask_colored.png` and `mask_palette.json` so mask labels can
be matched to the exported part colors.

## Python API

```python
from autopartgen import (
    GenerationOptions,
    generate_from_image,
    generate_from_image_and_mask,
    generate_from_mesh,
    load_pipeline,
)

pipeline = load_pipeline()  # uses the packaged autopartgen/configs/default.yaml
options = GenerationOptions(grid_size=512, seed=0, postprocess=True)

image_parts = generate_from_image(
    pipeline,
    "examples/image/apple_character/image.png",
    output_dir="outputs/apple_character",
    options=options,
)

mesh_parts = generate_from_mesh(
    pipeline,
    "examples/mesh/potted_flower/mesh.glb",
    output_dir="outputs/potted_flower_mesh",
    options=options,
)

masked_parts = generate_from_image_and_mask(
    pipeline,
    "examples/image_mesh_mask/robot/image.png",
    "examples/image_mesh_mask/robot/mask.png",
    mesh="examples/image_mesh_mask/robot/mesh.glb",
    output_dir="outputs/robot_image_mesh_mask",
    options=options,
)
```

To load the optional 4096-latent-token checkpoint from Python, pass the packaged
4096 config:

```python
pipeline = load_pipeline("autopartgen/configs/default_4096.yaml")
```

The package also provides `generate_from_image_and_mesh` for image-and-mesh
conditioning without an indexed mask.

## Runtime and Post-processing

Release inference loads `autopartgen/configs/default.yaml` by default through
`load_pipeline()`. This 2048-latent-token checkpoint is the main release path.
The optional 4096-latent-token finetuned checkpoint uses
`autopartgen/configs/default_4096.yaml`; pass it with `--config` in the CLI or
as the `config` argument to `load_pipeline()`. Runtime options can be passed
through CLI flags in `inference.py` or through `GenerationOptions` in Python.

Guidance is mode-specific. The current default 2048-latent release values are:

| Mode | Image CFG | Geometry CFG |
| --- | ---: | ---: |
| whole image-to-mesh stage | `whole_cfg_scale=7.0` | n/a |
| `image` | `0.0` | `2.0` |
| `mesh` | `0.0` | `2.0` |
| `image_mesh` | `0.0` | `2.0` |
| `image_mask` | `5.0` | `5.0` |
| `image_mesh_mask` | `5.0` | `5.0` |

`--image_cfg_scale`, `--geometry_cfg_scale`, `--mask_image_cfg_scale`,
`--mask_geometry_cfg_scale`, and `--whole_cfg_scale` only override the config for
that run; `autopartgen/configs/default.yaml` provides the default 2048-latent
values. The optional 4096-latent-token config uses the same scheduler and model
width, sets `dit.num_latents: 4096`, points to
`checkpoints/autopartgen_dit_4096.pth`, uses the VAE with
`latent_eval_downscale: 8`.

> [!NOTE]
> - Higher geometry guidance usually encourages more segments and accepted
>   parts, and can sometimes make parts sharper.
> - AutoPartGen is a generative model, so try a different `--seed` if one sample
>   does not work well.
> - To follow the input image more closely, increase the image CFG scale. To
>   follow the input mesh more closely, increase the geometry guidance scale.

`--grid_size` sets the final iso-surface resolution. It must be a power of two;
the default is `512`. Use `256` for faster lower-resolution checks.

`--no_post` disables per-part floater removal, simplification, final scene
cleanup, and final smoothing.

`--smooth_iters N` enables a final Taubin smoothing pass only when `N > 0`.
The default `--smooth_iters 0` keeps all other post-processing steps enabled
without running Taubin smoothing.

`--simplify_faces N` applies per-part quadric simplification. The default is
`50000`; set `--simplify_faces 0` to disable simplification. AutoPartGen tries
`fast-simplification`, then pymeshlab, then Open3D. If no simplification backend
is available, the original mesh is kept and a warning is emitted. The
fast-simplification aggression can be overridden with `APG_SIMPLIFY_AGG`
(default `1.0`).

`--isosurface_backend {auto,diso,skimage}` selects the iso-surface extraction
backend. The default `auto` uses `diso` (DiffDMC) when installed and otherwise
falls back to `skimage`. Use `diso` to require DiffDMC, which usually gives
cleaner watertight parts but needs the optional CUDA extension. Use `skimage` to
force the dependency-free marching-cubes fallback.

`--iou_threshold` defaults to `0.3`. Floater removal and simplification run only
after a candidate passes this duplicate check. The duplicate check uses
`--iou_grid_size 256` by default, and the sampled-surface fallback uses 500k
points.

`--seed N` controls diffusion sampling, mesh surface resampling, FPS start
points, IoU fallback sampling, and optional posterior/history noise. Re-running
the same command with the same seed, inputs, checkpoints, backend, and package
versions should produce the same outputs.

`--max_parts N` is a hard cap for image- or mesh-conditioned generation without
masks. The model can stop earlier when it predicts the end token. When masks are
provided, the number of part attempts follows the number of mask regions, so
progress is shown as `x/y` only in that mode.

`--no_remove_background` disables the default image background removal path.
Existing valid alpha masks are preserved even when background removal is enabled.

`--no_progress` disables stage logs and diffusion progress bars.

`--use_coarse_bbox` enables the coarse ROI bbox crop during mesh extraction.

`--no_mask_visualization` disables `mask_colored.png` and `mask_palette.json`
for indexed-mask inputs.

The first image-conditioned run downloads DINOv2-L/14 with registers through
`torch.hub`; subsequent runs use the local PyTorch cache.

## Examples

The `examples/` directory is grouped by inference scenario:

```text
image/<name>/       Image-only examples
mesh/<name>/        Mesh-only examples
image_mesh_mask/<name>/  Image, mesh, and mask examples
```

Each example directory contains only the input files for that scenario.

## License

The code and the released checkpoints (`autopartgen_dit.pth`,
`autopartgen_dit_4096.pth`, and `autopartgen_vae.pth`) are released under the
**FAIR Noncommercial Research License** (see [LICENSE](LICENSE)) —
noncommercial research use only.

## Acknowledgements

AutoPartGen builds upon several open-source projects and publicly released models. We sincerely thank their authors for their valuable contributions:

| Project | Use |
|---------|-----|
| [TripoSG](https://github.com/VAST-AI-Research/TripoSG) | VAE, DiT backbone, and image-to-3D pipeline structure |
| [HunyuanDiT](https://github.com/Tencent/HunyuanDiT) | Transformer blocks used by the TripoSG implementation |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Image features |
| [Diffusers](https://github.com/huggingface/diffusers) | Model and scheduler utilities |
| [Transformers](https://github.com/huggingface/transformers) | Background-removal model loading |
| [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) | Background removal |
| [DiffDMC / diso](https://github.com/SarahWeiii/diso) | Dual marching-cubes surface extraction |
| [TRELLIS](https://github.com/microsoft/TRELLIS) | Mesh post-processing utilities |

## Citation

```bibtex
@article{chen2025autopartgen,
  title={AutoPartGen: Autoregressive 3D Part Generation and Discovery},
  author={Minghao Chen and Jianyuan Wang and Roman Shapovalov and Tom Monnier
          and Hyunyoung Jung and Dilin Wang and Rakesh Ranjan and Iro Laina
          and Andrea Vedaldi},
  journal={arXiv preprint arXiv:2507.13346},
  year={2025}
}
```
