from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def save_attribution(
    path: str | Path,
    *,
    scores: Tensor,
    kl_shift: Tensor,
    curvature: Tensor,
    amplifier: Tensor,
    explained_class: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "explained_class": int(explained_class),
        "scores": scores.detach().cpu().tolist(),
        "kl_shift": kl_shift.detach().cpu().tolist(),
        "curvature": curvature.detach().cpu().tolist(),
        "amplifier": amplifier.detach().cpu().tolist(),
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_masks(path: str | Path, masks: Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(masks.detach().cpu(), path)
