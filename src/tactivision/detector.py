"""Tiled YOLO inference without shrinking an entire panorama into one small image."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    if tile <= 0 or length <= 0 or not 0 <= overlap < 1:
        raise ValueError("Positive tile size and overlap in [0,1) required")
    if length <= tile:
        return [0]
    step = max(1, round(tile * (1 - overlap)))
    return sorted(set([*range(0, length - tile + 1, step), length - tile]))


def suppress_duplicates(detections: list[Detection], iou: float = .4) -> list[Detection]:
    if not detections:
        return []
    boxes = [[d.xyxy[0], d.xyxy[1], d.xyxy[2] - d.xyxy[0], d.xyxy[3] - d.xyxy[1]]
             for d in detections]
    keep = cv2.dnn.NMSBoxes(boxes, [d.confidence for d in detections], 0.0, iou)
    return [detections[int(i)] for i in np.asarray(keep).reshape(-1)]


class BallDetector:
    def __init__(self, weights, confidence=.15, imgsz=1024,
                 tile_size=1024, overlap=.2, device="cpu", class_id=None, focus_region=None):
        if not 0 < confidence <= 1 or imgsz <= 0 or tile_size < 0 or not 0 <= overlap < 1:
            raise ValueError("Invalid confidence, image size, tile size or overlap")
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Local detector weights missing: {weights}. No automatic download performed.")
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        if class_id is None:
            names = self.model.names
            items = names.items() if isinstance(names, dict) else enumerate(names)
            matches = [int(i) for i, name in items if str(name).lower() in ("sports ball", "ball", "football", "soccer ball")]
            if len(matches) != 1:
                raise ValueError("Cannot uniquely identify ball class; provide --class-id")
            class_id = matches[0]
        if class_id not in range(len(self.model.names)):
            raise ValueError("Ball class id is outside the model class range")
        self.class_id = class_id
        self.confidence, self.imgsz = confidence, imgsz
        self.tile_size, self.overlap, self.device = tile_size, overlap, device
        self.focus_region = focus_region

    def detect(self, frame: np.ndarray) -> list[Detection]:
        original_height, original_width = frame.shape[:2]
        offset_x = offset_y = 0
        if self.focus_region is not None:
            low = np.floor(self.focus_region.vertices.min(axis=0) - 32).astype(int)
            high = np.ceil(self.focus_region.vertices.max(axis=0) + 32).astype(int)
            offset_x, offset_y = max(0, int(low[0])), max(0, int(low[1]))
            right, bottom = min(original_width, int(high[0])), min(original_height, int(high[1]))
            if offset_x >= right or offset_y >= bottom:
                raise ValueError("Playing region does not overlap the input image")
            frame = frame[offset_y:bottom, offset_x:right]
        height, width = frame.shape[:2]
        tile = self.tile_size or max(width, height)
        candidates = []
        # Process one tile at a time to bound GPU/CPU memory on 4K panoramas.
        for top in tile_starts(height, tile, self.overlap):
            for left in tile_starts(width, tile, self.overlap):
                crop = frame[top:top + tile, left:left + tile]
                result = self.model.predict(crop, conf=self.confidence, imgsz=self.imgsz,
                                            classes=[self.class_id], device=self.device,
                                            verbose=False)[0]
                if result.boxes is None:
                    continue
                for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = box.tolist()
                    xyxy = (max(float(offset_x), x1 + left + offset_x), max(float(offset_y), y1 + top + offset_y),
                            min(float(offset_x + width), x2 + left + offset_x),
                            min(float(offset_y + height), y2 + top + offset_y))
                    if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1] and np.isfinite([*xyxy, score]).all():
                        candidates.append(Detection(xyxy, float(score)))
        return suppress_duplicates(candidates)
