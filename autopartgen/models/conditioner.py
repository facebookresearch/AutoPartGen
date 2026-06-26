# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import warnings

import torch
import torch.nn.functional as F
from torchvision import transforms

def replace_with_dummy(x, drop_ids):
    drop_ids = drop_ids[(...,) + (None,) * (x.ndim - 1)]
    drop_ids = drop_ids.expand(x.shape)
    x = torch.where(drop_ids, 0, x)
    return x


class ImageConditioner(torch.nn.Module):
    def __init__(
        self,
        model_name,
        img_size=224,
        normalize=True,
        pretrained=True,
        norm_out="none",
        feature_layers=None,
    ):
        super().__init__()
        assert norm_out in ["layer_norm", "original", "none"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available.*")
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=pretrained,
                verbose=False,
            )
        self.img_size = img_size
        if normalize:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = lambda x: x
        self.norm_out = norm_out
        self.feature_layers = feature_layers
        self.eval()

    def forward(self, x):
        x = self.normalize(x)
        x = self.model.forward_features(x)["x_prenorm"]
        if self.norm_out == "layer_norm":
            x = F.layer_norm(x, x.shape[-1:])
        elif self.norm_out == "original":
            x = self.model.norm(x)
        return x
