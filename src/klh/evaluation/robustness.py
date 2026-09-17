from __future__ import annotations

import torch
from torch import Tensor


def _rankdata(values: Tensor) -> Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(values), device=values.device, dtype=torch.float32)
    return ranks


def spearman_rank(a: Tensor, b: Tensor) -> float:
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("inputs must be one-dimensional and have equal length")
    if len(a) < 2:
        return 1.0
    ra, rb = _rankdata(a.float()), _rankdata(b.float())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.linalg.vector_norm(ra) * torch.linalg.vector_norm(rb)
    if float(denom) == 0.0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float(torch.sum(ra * rb) / denom)


def jaccard_topk(a: Tensor, b: Tensor, k: int = 3) -> float:
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("inputs must be one-dimensional and have equal length")
    k = min(k, len(a))
    ia = set(torch.topk(a, k).indices.tolist())
    ib = set(torch.topk(b, k).indices.tolist())
    union = ia | ib
    return len(ia & ib) / len(union) if union else 1.0
