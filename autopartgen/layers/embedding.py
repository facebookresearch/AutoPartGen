# Copyright (c) Meta Platforms, Inc. and affiliates.

import math

import numpy as np
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, output_dim, num_freqs=128, max_period=10000):
        super().__init__()
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=num_freqs, dtype=torch.float32)
            / num_freqs
        )
        self.register_buffer("frequencies", freqs, persistent=False)
        self.linear = nn.Linear(num_freqs * 2, output_dim)

    def forward(self, x):
        x = x[:, None] * self.frequencies[None]
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = self.linear(x)
        return x


class CoordinateEmbedding(nn.Module):
    """Sinusoidal embeddings for [-1, 1] coordinate values."""

    def __init__(
        self,
        input_dim=3,
        output_dim=128,
        num_freqs=8,
        include_pi=False,
        include_input=True,
    ):
        super().__init__()
        self.include_input = include_input
        freqs = (2 ** torch.arange(num_freqs)).float()
        if include_pi:
            freqs = freqs * np.pi
        self.register_buffer("frequencies", freqs, persistent=False)

        dim = input_dim * (num_freqs * 2 + int(include_input))
        self.linear = nn.Linear(dim, output_dim)

    def forward(self, x):
        feats = (x[..., None] * self.frequencies).flatten(-2)
        if self.include_input:
            x = torch.cat([x, feats.sin(), feats.cos()], dim=2)
        else:
            x = torch.cat([feats.sin(), feats.cos()], dim=2)
        x = self.linear(x)
        return x
