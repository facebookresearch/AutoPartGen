# Copyright (c) Meta Platforms, Inc. and affiliates.

# References:
#   https://github.com/huggingface/diffusers/blob/v0.19.2/src/diffusers/utils/torch_utils.py

import logging
from typing import List, Optional, Tuple, Union

import torch

LOGGER = logging.getLogger(__name__)

TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def get_torch_dtype(precision):
    if precision is None:
        return torch.float32
    return TORCH_DTYPES[precision]


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = (
            generator.device.type
            if not isinstance(generator, list)
            else generator[0].device.type
        )
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                LOGGER.debug(
                    "Generator on 'cpu' but a tensor on %s was expected; creating on "
                    "'cpu' then moving to %s.",
                    device, device,
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {device} tensor from a generator of type {gen_device_type}."
            )

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(
            shape, generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    return latents


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )
