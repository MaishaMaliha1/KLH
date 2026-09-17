from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import AttributionConfig


@dataclass(frozen=True)
class PartAttribution:
    scores: Tensor
    kl_shift: Tensor
    curvature: Tensor
    amplifier: Tensor
    explained_class: int


def _l2_normalize(vector: Tensor, eps: float) -> Tensor:
    return vector / (torch.linalg.vector_norm(vector) + eps)


def _symmetric_kl(p: Tensor, q: Tensor, eps: float) -> Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    kl_pq = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
    kl_qp = torch.sum(q * (torch.log(q) - torch.log(p)), dim=-1)
    return 0.5 * (kl_pq + kl_qp)


class KLHessianAttributor:
    """Part-level attribution using symmetric KL shift and directional curvature."""

    def __init__(self, model: nn.Module, config: AttributionConfig | None = None):
        self.model = model.eval()
        self.config = config or AttributionConfig()
        self.config.validate()

    def _logits(self, image: Tensor) -> Tensor:
        output = self.model(image)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, Tensor) or output.ndim != 2:
            raise ValueError("model must return a [batch, classes] logits tensor")
        return output

    def _class_gradient(self, image: Tensor, class_index: int) -> Tensor:
        probe = image.detach().clone().requires_grad_(True)
        logits = self._logits(probe)
        scalar = logits[0, class_index]
        grad = torch.autograd.grad(scalar, probe, create_graph=False, retain_graph=False)[0]
        return grad.detach()

    @torch.enable_grad()
    def attribute(
        self,
        image: Tensor,
        masks: Tensor,
        class_index: int | None = None,
    ) -> PartAttribution:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("image must have shape [1, C, H, W]")
        if masks.ndim != 3:
            raise ValueError("masks must have shape [K, H, W]")
        if tuple(masks.shape[-2:]) != tuple(image.shape[-2:]):
            raise ValueError("mask spatial size must match image spatial size")

        cfg = self.config
        device = image.device
        masks = masks.to(device=device, dtype=image.dtype)
        masks = (masks > 0).to(image.dtype)

        with torch.no_grad():
            base_logits = self._logits(image)
            if class_index is None:
                class_index = int(base_logits.argmax(dim=1).item())
            p0 = torch.softmax(base_logits, dim=-1)

        g = self._class_gradient(image, class_index)

        scores: list[Tensor] = []
        shifts: list[Tensor] = []
        curvatures: list[Tensor] = []
        amplifiers: list[Tensor] = []

        for mask_2d in masks:
            mask = mask_2d.unsqueeze(0).unsqueeze(0).expand_as(image)
            v = _l2_normalize(mask * g, cfg.numerical_epsilon)

            grad_plus = self._class_gradient(image + cfg.delta * v, class_index)
            grad_minus = self._class_gradient(image - cfg.delta * v, class_index)
            hv = (grad_plus - grad_minus) / (2.0 * cfg.delta)

            h_hat = _l2_normalize(mask * hv, cfg.numerical_epsilon)
            u = _l2_normalize(
                v + cfg.curvature_blend * h_hat,
                cfg.numerical_epsilon,
            )

            with torch.no_grad():
                p_plus = torch.softmax(self._logits(image + cfg.epsilon * u), dim=-1)
                p_minus = torch.softmax(self._logits(image - cfg.epsilon * u), dim=-1)

            kl_shift = 0.5 * (
                _symmetric_kl(p0, p_plus, cfg.numerical_epsilon)
                + _symmetric_kl(p0, p_minus, cfg.numerical_epsilon)
            )
            curvature = torch.sum(v * hv)
            amplifier = 1.0 + cfg.curvature_weight * torch.clamp(curvature, min=0.0)
            score = kl_shift.squeeze(0) * amplifier

            scores.append(score.detach())
            shifts.append(kl_shift.squeeze(0).detach())
            curvatures.append(curvature.detach())
            amplifiers.append(amplifier.detach())

        return PartAttribution(
            scores=torch.stack(scores),
            kl_shift=torch.stack(shifts),
            curvature=torch.stack(curvatures),
            amplifier=torch.stack(amplifiers),
            explained_class=class_index,
        )
