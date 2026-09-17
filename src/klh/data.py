from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from PIL import Image

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: str | Path) -> Iterator[tuple[Path, Image.Image]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            with Image.open(path) as image:
                yield path, image.convert("RGB")
