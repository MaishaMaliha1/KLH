from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AttributionConfig:
    epsilon: float = 0.20
    delta: float = 0.06
    curvature_blend: float = 0.40
    curvature_weight: float = 1.0
    numerical_epsilon: float = 1.0e-12

    def validate(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        if self.curvature_blend < 0:
            raise ValueError("curvature_blend must be nonnegative")
        if self.curvature_weight < 0:
            raise ValueError("curvature_weight must be nonnegative")
        if self.numerical_epsilon <= 0:
            raise ValueError("numerical_epsilon must be positive")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("configuration root must be a mapping")
    return payload


def attribution_config(payload: dict[str, Any]) -> AttributionConfig:
    cfg = AttributionConfig(**payload.get("attribution", {}))
    cfg.validate()
    return cfg
