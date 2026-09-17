from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .attribution import KLHessianAttributor
from .config import attribution_config, load_yaml
from .io import save_attribution, save_masks
from .models import create_model
from .segmentation import AnimalPoseParts, DINOv2KMeansParts, FaceMeshParts, GridParts, VehicleParts


def _build_parts(payload: dict, device: str):
    cfg = payload.get("parts", {})
    backend = str(cfg.get("method", cfg.get("backend", "dino"))).lower()
    if backend in {"dino", "dinov2", "dinov2_kmeans"}:
        return DINOv2KMeansParts(
            count=int(cfg.get("count", 12)),
            min_area_ratio=float(cfg.get("min_area_ratio", 0.0025)),
            device=device,
        )
    if backend in {"face", "facemesh"}:
        return FaceMeshParts()
    if backend == "grid":
        return GridParts(8, 8)
    if backend in {"animal", "animalpose"}:
        return AnimalPoseParts(checkpoint=cfg.get("checkpoint"), device=device)
    if backend == "vehicle":
        return VehicleParts(
            count=int(cfg.get("count", 10)),
            detector=str(cfg.get("detector", "yolov8n.pt")),
            device=device,
            min_area_ratio=float(cfg.get("min_area_ratio", 0.0025)),
        )
    raise ValueError(f"unknown parts backend: {backend}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KL-Hessian part-level attribution")
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/attribution.json"))
    parser.add_argument("--masks", type=Path, default=None)
    parser.add_argument("--class-index", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = load_yaml(args.config)
    model_cfg = payload.get("model", "vit_b16")
    if isinstance(model_cfg, str):
        model_name = model_cfg
        device = str(payload.get("device", "cuda"))
        pretrained = True
    elif isinstance(model_cfg, dict):
        model_name = str(model_cfg.get("name", "vit_b16"))
        device = str(model_cfg.get("device", "cuda"))
        pretrained = bool(model_cfg.get("pretrained", True))
    else:
        raise TypeError("model configuration must be a model alias or mapping")

    bundle = create_model(model_name, pretrained=pretrained, device=device)
    model_device = next(bundle.model.parameters()).device

    image_pil = Image.open(args.image).convert("RGB")
    image_tensor = bundle.preprocess(image_pil).unsqueeze(0).to(model_device)

    if args.masks is not None:
        masks = torch.load(args.masks, map_location="cpu", weights_only=True)
    else:
        masks = _build_parts(payload, str(model_device))(image_pil)
    masks = torch.nn.functional.interpolate(
        masks[:, None].float(), size=image_tensor.shape[-2:], mode="nearest"
    )[:, 0]

    result = KLHessianAttributor(bundle.model, attribution_config(payload)).attribute(
        image_tensor, masks, args.class_index
    )
    save_attribution(
        args.output,
        scores=result.scores,
        kl_shift=result.kl_shift,
        curvature=result.curvature,
        amplifier=result.amplifier,
        explained_class=result.explained_class,
        metadata={"model": bundle.name, "image": str(args.image)},
    )
    mask_path = args.output.with_suffix(".masks.pt")
    save_masks(mask_path, masks)
    print(f"saved attribution to {args.output}")
    print(f"saved masks to {mask_path}")


if __name__ == "__main__":
    main()
