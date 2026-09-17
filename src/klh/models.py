from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    preprocess: Callable
    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    name: str


# Explicit model registry. We fail fast rather than silently swapping checkpoints.
_MODEL_REGISTRY: dict[str, str] = {
    "vit_b16": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "convnextv2_b": "convnextv2_base.fcmae_ft_in22k_in1k",
    "swinv2_b": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "dino_vit_b16": "vit_base_patch16_224.dino",
}


def supported_models() -> tuple[str, ...]:
    return tuple(_MODEL_REGISTRY)


def model_entry(alias: str) -> str:
    if alias not in _MODEL_REGISTRY:
        allowed = ", ".join(supported_models())
        raise ValueError(f"unknown model '{alias}'. Supported aliases: {allowed}")
    return _MODEL_REGISTRY[alias]


def create_model(
    name: str,
    *,
    pretrained: bool = True,
    device: str | torch.device = "cuda",
) -> ModelBundle:
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ImportError as exc:
        raise RuntimeError("timm is required to create the vision models") from exc

    entry = model_entry(name)
    available = set(timm.list_models(pretrained=pretrained))
    if entry not in available:
        raise RuntimeError(
            f"required timm model '{entry}' is not available in this environment. "
            "Install a compatible timm version rather than substituting a different checkpoint."
        )

    model = timm.create_model(entry, pretrained=pretrained).eval()
    data_cfg = resolve_model_data_config(model)
    transform = create_transform(**data_cfg, is_training=False)
    size = data_cfg.get("input_size", (3, 224, 224))
    mean = tuple(float(v) for v in data_cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(v) for v in data_cfg.get("std", (0.229, 0.224, 0.225)))

    actual_device = torch.device(device)
    if actual_device.type == "cuda" and not torch.cuda.is_available():
        actual_device = torch.device("cpu")
    model = model.to(actual_device)

    return ModelBundle(
        model=model,
        preprocess=transform,
        input_size=(int(size[-2]), int(size[-1])),
        mean=mean,
        std=std,
        name=entry,
    )
