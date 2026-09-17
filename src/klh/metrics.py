from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms.functional import gaussian_blur


@dataclass(frozen=True)
class FaithfulnessScores:
    insertion: float
    deletion: float
    aopc: float


def _class_probability(model: nn.Module, image: Tensor, class_index: int) -> Tensor:
    output = model(image)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return torch.softmax(output, dim=-1)[0, class_index]


def _trapz_mean(values: list[Tensor]) -> Tensor:
    stacked = torch.stack(values)
    if stacked.numel() < 2:
        return stacked.mean()
    return torch.trapezoid(stacked, dx=1.0) / (stacked.numel() - 1)


@torch.no_grad()
def part_faithfulness(
    model: nn.Module,
    image: Tensor,
    baseline: Tensor,
    masks: Tensor,
    scores: Tensor,
    class_index: int,
) -> FaithfulnessScores:
    order = torch.argsort(scores, descending=True)
    insertion = baseline.clone()
    deletion = image.clone()
    p_original = _class_probability(model, image, class_index)
    ins_curve = [_class_probability(model, insertion, class_index)]
    del_curve = [_class_probability(model, deletion, class_index)]
    drops = []

    for idx in order.tolist():
        mask = masks[idx].to(image.device, image.dtype)[None, None]
        insertion = insertion * (1.0 - mask) + image * mask
        deletion = deletion * (1.0 - mask) + baseline * mask
        ins_curve.append(_class_probability(model, insertion, class_index))
        p_del = _class_probability(model, deletion, class_index)
        del_curve.append(p_del)
        drops.append(torch.abs(p_original - p_del))

    return FaithfulnessScores(
        insertion=float(_trapz_mean(ins_curve)),
        deletion=float(_trapz_mean(del_curve)),
        aopc=float(torch.stack(drops).mean()) if drops else 0.0,
    )


def part_scores_to_pixel_map(scores: Tensor, masks: Tensor, smoothing_kernel: int = 5) -> Tensor:
    if smoothing_kernel < 1 or smoothing_kernel % 2 == 0:
        raise ValueError("smoothing_kernel must be a positive odd integer")
    score_map = torch.zeros_like(masks[0], dtype=scores.dtype, device=scores.device)
    weight_map = torch.zeros_like(score_map)
    for score, mask in zip(scores, masks.to(scores.device), strict=True):
        score_map = score_map + score * mask
        weight_map = weight_map + mask
    score_map = score_map / weight_map.clamp_min(1.0)
    if smoothing_kernel > 1:
        pad = smoothing_kernel // 2
        score_map = F.avg_pool2d(
            score_map[None, None],
            kernel_size=smoothing_kernel,
            stride=1,
            padding=pad,
        )[0, 0]
    return score_map


@torch.no_grad()
def pixel_faithfulness(
    model: nn.Module,
    image: Tensor,
    baseline: Tensor,
    attribution_map: Tensor,
    class_index: int,
    steps: int = 50,
) -> FaithfulnessScores:
    if steps < 1:
        raise ValueError("steps must be positive")
    ranking = torch.argsort(attribution_map.flatten(), descending=True)
    chunks = torch.tensor_split(ranking, min(steps, ranking.numel()))

    insertion = baseline.clone()
    deletion = image.clone()
    p_original = _class_probability(model, image, class_index)
    ins_curve = [_class_probability(model, insertion, class_index)]
    del_curve = [_class_probability(model, deletion, class_index)]
    drops = []

    flat_ins = insertion.view(1, image.shape[1], -1)
    flat_del = deletion.view(1, image.shape[1], -1)
    flat_img = image.view(1, image.shape[1], -1)
    flat_base = baseline.view(1, image.shape[1], -1)

    for chunk in chunks:
        flat_ins[:, :, chunk] = flat_img[:, :, chunk]
        flat_del[:, :, chunk] = flat_base[:, :, chunk]
        ins_curve.append(_class_probability(model, insertion, class_index))
        p_del = _class_probability(model, deletion, class_index)
        del_curve.append(p_del)
        drops.append(p_original - p_del)

    return FaithfulnessScores(
        insertion=float(_trapz_mean(ins_curve)),
        deletion=float(_trapz_mean(del_curve)),
        aopc=float(torch.stack(drops).mean()) if drops else 0.0,
    )


def blurred_baseline(image_01: Tensor, sigma: float = 11.0) -> Tensor:
    if image_01.ndim != 4 or image_01.shape[0] != 1:
        raise ValueError("image_01 must have shape [1, C, H, W]")
    radius = max(1, round(3.0 * sigma))
    kernel = 2 * radius + 1
    return gaussian_blur(image_01, [kernel, kernel], [sigma, sigma])
