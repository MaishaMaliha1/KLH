from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from klh.attribution import KLHessianAttributor
from klh.config import attribution_config, load_yaml
from klh.models import create_model
from klh.segmentation import DINOv2KMeansParts, FaceMeshParts, GridParts, load_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KL-Hessian attribution on one image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--model", default="vit_b16")
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--parts", choices=["dino", "face", "grid"], default="dino")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_yaml(args.config)
    cfg = attribution_config(payload)
    bundle = create_model(args.model, device=args.device)
    pil = Image.open(args.image).convert("RGB")
    image = bundle.preprocess(pil).unsqueeze(0).to(next(bundle.model.parameters()).device)

    if args.masks:
        masks = load_masks(args.masks)
    elif args.parts == "face":
        masks = FaceMeshParts()(pil)
    elif args.parts == "grid":
        masks = GridParts()(pil)
    else:
        count = int(payload.get("parts", {}).get("count", 12))
        min_area = float(payload.get("parts", {}).get("min_area_ratio", 0.0025))
        masks = DINOv2KMeansParts(count=count, min_area_ratio=min_area, device=args.device)(pil)

    masks = torch.nn.functional.interpolate(
        masks[:, None], size=image.shape[-2:], mode="nearest"
    )[:, 0]
    result = KLHessianAttributor(bundle.model, cfg).attribute(image, masks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": bundle.name,
                "explained_class": result.explained_class,
                "scores": result.scores.cpu().tolist(),
                "kl_shift": result.kl_shift.cpu().tolist(),
                "curvature": result.curvature.cpu().tolist(),
                "amplifier": result.amplifier.cpu().tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
