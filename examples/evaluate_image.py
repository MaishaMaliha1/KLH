from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

from klh.attribution import KLHessianAttributor
from klh.config import AttributionConfig
from klh.metrics import part_faithfulness, part_scores_to_pixel_map, pixel_faithfulness
from klh.models import create_model
from klh.segmentation import DINOv2KMeansParts


def _raw_tensor(image: Image.Image, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    image = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _normalize(image_01: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=image_01.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=image_01.device).view(1, 3, 1, 1)
    return (image_01 - mean_t) / std_t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="vit_b16")
    args = parser.parse_args()

    bundle = create_model(args.model)
    device = next(bundle.model.parameters()).device
    image_pil = Image.open(args.image).convert("RGB")
    image = bundle.preprocess(image_pil).unsqueeze(0).to(device)

    parts = DINOv2KMeansParts(count=12, device=str(device))(image_pil)
    parts = torch.nn.functional.interpolate(
        parts[:, None], size=image.shape[-2:], mode="nearest"
    )[:, 0].to(device)

    result = KLHessianAttributor(bundle.model, AttributionConfig()).attribute(image, parts)
    pixel_map = part_scores_to_pixel_map(result.scores, parts, smoothing_kernel=5)

    raw = _raw_tensor(image_pil, bundle.input_size, device)
    radius = max(1, int(round(3.0 * 11.0)))
    blurred = gaussian_blur(raw, [2 * radius + 1, 2 * radius + 1], [11.0, 11.0])
    baseline = _normalize(blurred, bundle.mean, bundle.std)

    part_scores = part_faithfulness(
        bundle.model, image, baseline, parts, result.scores, result.explained_class
    )
    pixel_scores = pixel_faithfulness(
        bundle.model, image, baseline, pixel_map, result.explained_class, steps=50
    )
    print("part:", part_scores)
    print("pixel:", pixel_scores)


if __name__ == "__main__":
    main()
