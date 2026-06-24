# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import os
import sys

import omegaconf
from PIL import Image


def get_simple_logger(name, level=logging.DEBUG):
    formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    while logger.handlers:
        logger.handlers.pop()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def recursive_merge_configs(
    config, default_key="__default__", base_config_path=None, root_path=None
):
    if default_key in config:
        if root_path is not None:
            config_path = os.path.join(root_path, config[default_key])
        else:
            config_path = config[default_key]
        default_config = omegaconf.OmegaConf.load(config_path)
        del config[default_key]
        merged_config = omegaconf.OmegaConf.merge(default_config, config)
        return recursive_merge_configs(
            merged_config, default_key, base_config_path, root_path
        )
    else:
        if base_config_path is not None:
            if root_path is not None:
                base_config_path = os.path.join(root_path, base_config_path)
            default_config = omegaconf.OmegaConf.load(base_config_path)
            return omegaconf.OmegaConf.merge(default_config, config)
        else:
            return config


def load_config(
    path,
    default_key="__default__",
    base_config_path=None,
    root_path=None,
    merge_cli=False,
):
    """
    Load an OmegaConf file and recursively merge other configs when __default__ is a key
    Args:
        path (str): Path to the main OmegaConf file.
        default_key (str): Key to recursively load default config form
        base_config_path (str): Path to the base OmegaConf file.
        root_path (str): Absolute path to the root of the config folder.
        merge_cli (bool): Merge OmegaConf CLI dotlist overrides into the file config.
    Returns:
        dict: The merged and resolved configuration as a dictionary.
    """
    if root_path is not None:
        path = os.path.join(root_path, path)
    config = omegaconf.OmegaConf.load(path)
    merged_config = recursive_merge_configs(config, default_key, base_config_path, root_path)
    if merge_cli:
        merged_config = omegaconf.OmegaConf.merge(
            merged_config, omegaconf.OmegaConf.from_cli()
        )

    return omegaconf.OmegaConf.to_container(merged_config, resolve=True)


def makedirs(dir_path, exist_ok=True):
    os.makedirs(dir_path, exist_ok=exist_ok)
    return dir_path


def load_image(path):
    """
    Load an image or binary mask from the given path.

    Args:
        path (str): Path to the image or mask file

    Returns:
        PIL.Image: Loaded image or mask
    """
    image = Image.open(path)

    # Masks (grayscale / palette / binary) pass through unchanged. For conditioning
    # images, preserve alpha so foreground crop/compositing downstream can match
    # the model's expected preprocessing. Plain RGB images are returned as RGB.
    if image.mode in ["L", "P", "1"]:
        return image
    if image.mode in ("RGBA", "LA"):
        return image.convert("RGBA")
    return image.convert("RGB")
