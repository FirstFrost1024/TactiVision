"""The playing-area polygon in the actual image, from official boundary keypoints."""

import json
from pathlib import Path

import cv2
import numpy as np


class PitchRegion:
    def __init__(self, vertices):
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
        if len(vertices) < 4 or not np.isfinite(vertices).all() or abs(cv2.contourArea(vertices)) < 1:
            raise ValueError("Playing region must have at least four finite, non-collinear vertices")
        self.vertices = vertices

    @classmethod
    def load(cls, path, calibration, space="distorted"):
        points = json.loads(Path(path).read_text())
        pairs = [(tuple(map(float, key.strip("() ").split(","))), pixel) for key, pixel in points.items()]
        L, W = calibration.pitch_length, calibration.pitch_width
        close = lambda a, b: abs(a - b) < 1e-6
        # Traverse the four sides in world-coordinate order. Unlike a convex hull,
        # this retains the curved touchlines in the original fisheye panorama.
        sides = [sorted([(w, p) for w, p in pairs if close(w[1], 0)], key=lambda q: q[0][0]),
                 sorted([(w, p) for w, p in pairs if close(w[0], L) and w[1] > 0], key=lambda q: q[0][1]),
                 sorted([(w, p) for w, p in pairs if close(w[1], W) and w[0] < L], key=lambda q: -q[0][0]),
                 sorted([(w, p) for w, p in pairs if close(w[0], 0) and 0 < w[1] < W], key=lambda q: -q[0][1])]
        for corner in ((0, 0), (L, 0), (L, W), (0, W)):
            if not any(close(w[0], corner[0]) and close(w[1], corner[1]) for w, _ in pairs):
                raise ValueError(f"Official pitch corner is missing: {corner}")
        vertices = np.array([p for side in sides for _, p in side], dtype=float)
        if space == "undistorted":
            vertices = calibration.undistort(vertices)
        elif space != "distorted":
            raise ValueError("Unknown image space")
        return cls(vertices)

    def distance(self, point):
        return float(cv2.pointPolygonTest(self.vertices, tuple(map(float, point)), True))
