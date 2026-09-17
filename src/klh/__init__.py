"""KL-Hessian attribution package."""

from .attribution import KLHessianAttributor, PartAttribution
from .config import AttributionConfig
from .models import ModelBundle, create_model

__all__ = [
    "AttributionConfig",
    "KLHessianAttributor",
    "ModelBundle",
    "PartAttribution",
    "create_model",
]

__version__ = "1.0.0"
