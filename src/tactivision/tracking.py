"""Conservative single-ball association in pitch metres; never invent missed positions."""

from dataclasses import dataclass

import numpy as np

from .detector import Detection


@dataclass
class Candidate:
    detection: Detection
    pitch_xy: np.ndarray


class BallAssociator:
    def __init__(self, max_speed=45.0, gate_slack=2.0, reset_after=.4,
                 region=None, off_field_grace=1.0, static_seconds=1.0):
        if not all(np.isfinite(x) and x >= 0 for x in (max_speed, gate_slack, reset_after, off_field_grace, static_seconds)):
            raise ValueError("Association limits must be nonnegative and finite")
        self.max_speed, self.gate_slack, self.reset_after = max_speed, gate_slack, reset_after
        self.last_time = None
        self.last_xy = None
        self.track_id = 0
        self.region = region
        self.off_field_grace, self.static_seconds = off_field_grace, static_seconds
        self.last_inside_time = None
        self.outside_dwell = []
        self.excluded_outside = 0
        self.excluded_static = 0

    def _playing_candidates(self, candidates, timestamp, active):
        if self.region is None:
            return candidates
        # Observe off-field candidates even when they were not selected. Memory is
        # local to this video and expires after a detection disappears for 0.2 s.
        self.outside_dwell = [entry for entry in self.outside_dwell if timestamp - entry["last"] <= .2]
        accepted = []
        for candidate in candidates:
            centre = np.array(candidate.detection.center)
            if self.region.distance(centre) >= 0:
                accepted.append(candidate)
                continue
            entry = next((entry for entry in self.outside_dwell if np.linalg.norm(centre - entry["anchor"]) <= 4), None)
            if entry is None:
                entry = dict(anchor=centre, first=timestamp, last=timestamp)
                self.outside_dwell.append(entry)
            entry["last"] = timestamp
            stationary = self.static_seconds > 0 and timestamp - entry["first"] >= self.static_seconds
            if stationary:
                self.excluded_static += 1
            elif (active and self.last_inside_time is not None
                  and timestamp - self.last_inside_time <= self.off_field_grace):
                # An established ball may leave the field. It must still pass the
                # ordinary distance gate; an off-field candidate cannot start a track.
                accepted.append(candidate)
                continue
            self.excluded_outside += 1
        return accepted

    def select(self, candidates: list[Candidate], timestamp: float) -> Candidate | None:
        candidates = [c for c in candidates if np.isfinite(c.pitch_xy).all()]
        if not candidates:
            return None
        active = self.last_time is not None and timestamp - self.last_time <= self.reset_after
        candidates = self._playing_candidates(candidates, timestamp, active)
        if not candidates:
            return None
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
        if self.region is not None and self.region.distance(selected.detection.center) >= 0:
            self.last_inside_time = timestamp
        return selected
