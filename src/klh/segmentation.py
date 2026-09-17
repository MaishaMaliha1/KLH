from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from torch import Tensor


class PartGenerator(Protocol):
    def __call__(self, image: Image.Image | Tensor) -> Tensor: ...


def _image_tensor(image: Image.Image | Tensor) -> Tensor:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)
    tensor = image.detach().float().cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("batched image must contain exactly one image")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError("image tensor must have shape [C, H, W]")
    return tensor


def _pil_image(image: Image.Image | Tensor) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    tensor = _image_tensor(image)
    arr = tensor.permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")


def make_disjoint(masks: Tensor) -> Tensor:
    """Convert possibly overlapping binary masks into a non-overlapping partition.

    Earlier masks take precedence. Empty masks are removed.
    """
    if masks.ndim != 3:
        raise ValueError("masks must have shape [K, H, W]")
    occupied = torch.zeros_like(masks[0], dtype=torch.bool)
    parts: list[Tensor] = []
    for raw in masks:
        current = raw.bool() & ~occupied
        if current.any():
            parts.append(current.float())
            occupied |= current
    if not parts:
        raise RuntimeError("no non-empty masks remain after overlap removal")
    return torch.stack(parts)


@dataclass(frozen=True)
class GridParts:
    rows: int = 8
    cols: int = 8

    def __call__(self, image: Image.Image | Tensor) -> Tensor:
        if isinstance(image, Image.Image):
            h, w = image.height, image.width
        else:
            h, w = int(image.shape[-2]), int(image.shape[-1])
        masks = []
        y_edges = np.linspace(0, h, self.rows + 1, dtype=int)
        x_edges = np.linspace(0, w, self.cols + 1, dtype=int)
        for r in range(self.rows):
            for c in range(self.cols):
                mask = torch.zeros((h, w), dtype=torch.float32)
                mask[y_edges[r] : y_edges[r + 1], x_edges[c] : x_edges[c + 1]] = 1.0
                masks.append(mask)
        return torch.stack(masks)


@dataclass
class DINOv2KMeansParts:
    count: int = 12
    min_area_ratio: float = 0.0025
    device: str = "cuda"
    model_name: str = "dinov2_vitb14"

    def __post_init__(self) -> None:
        if self.count < 2:
            raise ValueError("count must be at least 2")
        if not 0 <= self.min_area_ratio < 1:
            raise ValueError("min_area_ratio must be in [0, 1)")
        actual = torch.device(self.device)
        if actual.type == "cuda" and not torch.cuda.is_available():
            actual = torch.device("cpu")
        self._device = actual
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            self._model = self._model.to(self._device).eval()
        return self._model

    @staticmethod
    def _deterministic_init(tokens: np.ndarray, count: int) -> np.ndarray:
        """Data-derived farthest-point initialization for stable clustering."""
        center = tokens.mean(axis=0, keepdims=True)
        first = int(np.argmax(np.linalg.norm(tokens - center, axis=1)))
        chosen = [first]
        min_dist = np.linalg.norm(tokens - tokens[first], axis=1)
        for _ in range(1, count):
            idx = int(np.argmax(min_dist))
            chosen.append(idx)
            min_dist = np.minimum(min_dist, np.linalg.norm(tokens - tokens[idx], axis=1))
        return tokens[np.asarray(chosen)]

    @torch.no_grad()
    def __call__(self, image: Image.Image | Tensor) -> Tensor:
        tensor = _image_tensor(image)
        out_h, out_w = int(tensor.shape[-2]), int(tensor.shape[-1])
        resized = F.interpolate(
            tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized = ((resized - mean) / std).to(self._device)

        features = self._load_model().forward_features(normalized)
        tokens = features.get("x_norm_patchtokens")
        if tokens is None:
            raise RuntimeError("DINOv2 backbone did not expose patch tokens")
        tokens_np = tokens[0].float().cpu().numpy()

        grid_side = round(tokens_np.shape[0] ** 0.5)
        if grid_side * grid_side != tokens_np.shape[0]:
            raise RuntimeError("patch token count is not a square grid")
        if self.count > tokens_np.shape[0]:
            raise ValueError("part count exceeds available patch tokens")

        init = self._deterministic_init(tokens_np, self.count)
        labels = KMeans(n_clusters=self.count, init=init, n_init=1).fit_predict(tokens_np)
        label_map = torch.from_numpy(labels.reshape(grid_side, grid_side)).long()
        label_map = F.interpolate(
            label_map[None, None].float(), size=(out_h, out_w), mode="nearest"
        )[0, 0].long()

        masks = []
        min_area = self.min_area_ratio * out_h * out_w
        for idx in range(self.count):
            mask = (label_map == idx).float()
            if float(mask.sum()) >= min_area:
                masks.append(mask)
        if not masks:
            raise RuntimeError("all generated parts were removed by the area threshold")
        return torch.stack(masks)


class FaceMeshParts:
    """Face-region masks built from MediaPipe Face Mesh landmarks."""

    _REGIONS: ClassVar[dict[str, list[int]]] = {
        "left_eye": [33, 160, 158, 133, 153, 144],
        "right_eye": [362, 385, 387, 263, 373, 380],
        "left_brow": [70, 63, 105, 66, 107],
        "right_brow": [336, 296, 334, 293, 300],
        "nose": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2],
        "upper_lip": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        "lower_lip": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        "left_cheek": [50, 101, 205, 187, 123, 116],
        "right_cheek": [280, 330, 425, 411, 352, 345],
        "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
        "chin": [152, 148, 176, 149, 150, 136, 172, 58, 132],
        "left_face": [234, 93, 132, 58, 172, 136, 150, 149, 176],
        "right_face": [454, 323, 361, 288, 397, 365, 379, 378, 400],
        "center_face": [168, 197, 5, 4, 1, 2, 164, 0],
    }

    def __call__(self, image: Image.Image | Tensor) -> Tensor:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("FaceMesh parts require mediapipe and opencv-python") from exc

        arr = np.asarray(_pil_image(image))
        h, w = arr.shape[:2]
        mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        result = mesh.process(arr)
        mesh.close()
        if not result.multi_face_landmarks:
            raise RuntimeError("no face landmarks were detected")
        landmarks = result.multi_face_landmarks[0].landmark

        masks = []
        for indices in self._REGIONS.values():
            polygon = np.array(
                [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
                dtype=np.int32,
            )
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(polygon), 1)
            masks.append(torch.from_numpy(mask).float())
        return make_disjoint(torch.stack(masks))


@dataclass
class AnimalPoseParts:
    """Generate coarse animal regions from OpenPifPaf/AnimalPose keypoints."""

    checkpoint: str | None = None
    device: str = "cuda"
    point_radius_ratio: float = 0.035

    def __call__(self, image: Image.Image | Tensor) -> Tensor:
        try:
            import cv2
            import openpifpaf
        except ImportError as exc:
            raise RuntimeError("animal parts require openpifpaf and opencv-python") from exc

        kwargs = {}
        if self.checkpoint:
            kwargs["checkpoint"] = self.checkpoint
        predictor = openpifpaf.Predictor(**kwargs)
        pil = _pil_image(image)
        predictions, _, _ = predictor.pil_image(pil)
        if not predictions:
            raise RuntimeError("no animal pose was detected")

        data = np.asarray(predictions[0].data)
        if data.ndim != 2 or data.shape[1] < 3:
            raise RuntimeError("unexpected OpenPifPaf keypoint format")
        visible = data[:, 2] > 0
        points = data[visible, :2]
        if len(points) < 3:
            raise RuntimeError("too few visible animal keypoints")

        h, w = pil.height, pil.width
        radius = max(2, round(min(h, w) * self.point_radius_ratio))
        canvas = np.zeros((h, w), dtype=np.int32)
        for idx, (x, y) in enumerate(points, start=1):
            cv2.circle(canvas, (round(x), round(y)), radius, idx, thickness=-1)

        # Group adjacent keypoint neighborhoods into coarse semantic regions by spatial position.
        x_mid = float(np.median(points[:, 0]))
        y_lo = float(np.quantile(points[:, 1], 0.33))
        y_hi = float(np.quantile(points[:, 1], 0.66))
        groups = [np.zeros((h, w), dtype=np.uint8) for _ in range(6)]
        for idx, (x, y) in enumerate(points, start=1):
            if y <= y_lo:
                group = 0  # head / upper structure
            elif y >= y_hi and x < x_mid:
                group = 3  # lower-left limb structure
            elif y >= y_hi:
                group = 4  # lower-right limb structure
            elif x < x_mid:
                group = 1  # left body structure
            else:
                group = 2  # right body structure
            groups[group][canvas == idx] = 1

        hull = cv2.convexHull(points.astype(np.int32))
        body = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(body, hull, 1)
        covered = np.clip(sum(groups), 0, 1)
        groups[5] = np.clip(body - covered, 0, 1).astype(np.uint8)
        masks = [torch.from_numpy(m).float() for m in groups if m.any()]
        if not masks:
            raise RuntimeError("animal parser produced no usable parts")
        return make_disjoint(torch.stack(masks))


@dataclass
class VehicleParts:
    """Generate vehicle regions from YOLOv8 detection and DINOv2 token clustering."""

    count: int = 10
    detector: str = "yolov8n.pt"
    device: str = "cuda"
    min_area_ratio: float = 0.0025

    def __call__(self, image: Image.Image | Tensor) -> Tensor:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("vehicle parts require ultralytics") from exc

        pil = _pil_image(image)
        arr = np.asarray(pil)
        model = YOLO(self.detector)
        result = model.predict(arr, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            raise RuntimeError("YOLOv8 did not detect a vehicle")

        # COCO vehicle classes: bicycle, car, motorcycle, bus, train, truck.
        vehicle_ids = {1, 2, 3, 5, 6, 7}
        candidates = []
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls in vehicle_ids:
                xyxy = box.xyxy[0].detach().cpu().numpy()
                area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
                candidates.append((area, xyxy))
        if not candidates:
            raise RuntimeError("YOLOv8 found objects but no supported vehicle class")
        _, xyxy = max(candidates, key=lambda item: item[0])
        x1, y1, x2, y2 = [round(v) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(pil.width, x2), min(pil.height, y2)
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError("invalid vehicle bounding box")

        crop = pil.crop((x1, y1, x2, y2))
        crop_parts = DINOv2KMeansParts(
            count=self.count,
            min_area_ratio=self.min_area_ratio,
            device=self.device,
        )(crop)
        masks = torch.zeros((crop_parts.shape[0], pil.height, pil.width), dtype=torch.float32)
        masks[:, y1:y2, x1:x2] = crop_parts
        return make_disjoint(masks)


def load_masks(path: str | Path) -> Tensor:
    masks = torch.load(Path(path), map_location="cpu")
    if not isinstance(masks, Tensor) or masks.ndim != 3:
        raise ValueError("stored masks must be a tensor with shape [K, H, W]")
    return (masks > 0).float()
