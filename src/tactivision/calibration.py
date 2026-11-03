"""Use SoccerTrack-v2's released fisheye model and pitch-to-image homography.

Only numeric NPZ members are read. Upstream pose arrays contain Python objects;
loading them with allow_pickle=True is neither necessary nor safe.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    result = np.full((len(points), 2), np.nan)
    valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 2]) > 1e-12)
    result[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:]
    return result


@dataclass
class Calibration:
    K: np.ndarray
    D: np.ndarray
    Knew: np.ndarray
    pitch_to_image: np.ndarray
    pitch_length: float = 105.0
    pitch_width: float = 68.0

    def __post_init__(self):
        for name in ("K", "Knew", "pitch_to_image"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3, 3) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 3x3 matrix")
            if np.linalg.matrix_rank(value) != 3:
                raise ValueError(f"{name} must be invertible")
            setattr(self, name, value)
        self.D = np.asarray(self.D, dtype=np.float64).reshape(-1, 1)
        if self.D.shape != (4, 1) or not np.isfinite(self.D).all():
            raise ValueError("SoccerTrack-v2 fisheye D must contain four finite coefficients")
        if not (np.isfinite(self.pitch_length) and np.isfinite(self.pitch_width)
                and self.pitch_length > 0 and self.pitch_width > 0):
            raise ValueError("Pitch dimensions must be positive and finite")
        self.image_to_pitch = np.linalg.inv(self.pitch_to_image)

    @classmethod
    def load(cls, directory: Path | str, match: str, **kwargs):
        directory = Path(directory)
        with np.load(directory / f"{match}_camera_intrinsics.npz", allow_pickle=False) as data:
            K, D, Knew = (data[key].copy() for key in ("K", "D", "Knew"))
        H = np.load(directory / f"{match}_homography.npy", allow_pickle=False)
        return cls(K, D, Knew, H, **kwargs)

    def undistort(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if len(pixels) == 0:
            return np.empty((0, 2))
        return cv2.fisheye.undistortPoints(pixels, self.K, self.D, P=self.Knew).reshape(-1, 2)

    def image_to_ground(self, pixels: np.ndarray, space: str = "distorted") -> np.ndarray:
        if space not in ("distorted", "undistorted"):
            raise ValueError("space must be distorted or undistorted")
        points = self.undistort(pixels) if space == "distorted" else pixels
        return transform(points, self.image_to_pitch)

    def ground_to_image(self, points: np.ndarray, space: str = "distorted") -> np.ndarray:
        if space not in ("distorted", "undistorted"):
            raise ValueError("space must be distorted or undistorted")
        undistorted = transform(points, self.pitch_to_image)
        if space == "undistorted" or len(undistorted) == 0:
            return undistorted
        rays = np.column_stack((undistorted, np.ones(len(undistorted)))) @ np.linalg.inv(self.Knew).T
        pixels, _ = cv2.fisheye.projectPoints(
            rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D
        )
        return pixels.reshape(-1, 2)

    def centered(self, points: np.ndarray) -> np.ndarray:
        """Centre origin, x right, y toward the far sideline (up in a pitch map)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return np.column_stack((points[:, 0] - self.pitch_length / 2,
                                self.pitch_width / 2 - points[:, 1]))

    def check_keypoints(self, path: Path | str) -> dict:
        data = json.loads(Path(path).read_text())
        pitch = np.array([list(map(float, key.strip("() ").split(","))) for key in data])
        pixels = np.array(list(data.values()), dtype=np.float64)
        if pitch.ndim != 2 or pitch.shape[1] != 2 or len(pitch) < 4:
            raise ValueError("Expected at least four SoccerTrack-v2 keypoint correspondences")
        reconstructed = self.image_to_ground(pixels)
        projected = self.ground_to_image(pitch)
        errors_m = np.linalg.norm(reconstructed - pitch, axis=1)
        errors_px = np.linalg.norm(projected - pixels, axis=1)
        if not np.isfinite(errors_m).all() or not np.isfinite(errors_px).all():
            raise ValueError("Calibration produces non-finite keypoint projections")
        return {
            "keypoint_count": len(pitch),
            "pitch_error_m": self._stats(errors_m),
            "distorted_reprojection_error_px": self._stats(errors_px),
            "note": ("Residuals between supplied keypoints and the K/D/Knew plus homography chain. "
                     "These are not ball localisation accuracy measurements and do not, by themselves, "
                     "establish that the official calibration is invalid or requires recalibration."),
            "keypoints": [dict(pitch_xy_m=w.tolist(), image_uv_px=p.tolist(),
                               reconstructed_xy_m=q.tolist(), error_m=float(e))
                          for w, p, q, e in zip(pitch, pixels, reconstructed, errors_m)],
        }

    @staticmethod
    def _stats(values):
        return {"rmse": float(np.sqrt(np.mean(values ** 2))),
                "median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
                "max": float(np.max(values))}
