"""Conservative single-ball association in pitch metres; never invent missed positions."""

from dataclasses import dataclass

import numpy as np

from .detector import Detection


@dataclass
class Candidate:
    detection: Detection
    pitch_xy: np.ndarray


class BallAssociator:
    def __init__(self, max_speed=45.0, gate_slack=2.0, reset_after=.4):
        if not all(np.isfinite(x) and x >= 0 for x in (max_speed, gate_slack, reset_after)):
            raise ValueError("Association limits must be nonnegative and finite")
        self.max_speed, self.gate_slack, self.reset_after = max_speed, gate_slack, reset_after
        self.last_time = None
        self.last_xy = None
        self.track_id = 0

    def select(self, candidates: list[Candidate], timestamp: float) -> Candidate | None:
        candidates = [c for c in candidates if np.isfinite(c.pitch_xy).all()]
        if not candidates:
            return None
        active = self.last_time is not None and timestamp - self.last_time <= self.reset_after
        if active:
            dt = timestamp - self.last_time
            if dt <= 0:
                raise ValueError("Timestamps must increase")
            gate = self.max_speed * dt + self.gate_slack
            candidates = [c for c in candidates if np.linalg.norm(c.pitch_xy - self.last_xy) <= gate]
            if not candidates:
                return None
            selected = max(candidates, key=lambda c: c.detection.confidence
                           - .25 * np.linalg.norm(c.pitch_xy - self.last_xy) / max(gate, 1e-6))
        else:
            selected = max(candidates, key=lambda c: c.detection.confidence)
            self.track_id += 1
        self.last_xy = selected.pitch_xy.copy()
        self.last_time = timestamp
        return selected
