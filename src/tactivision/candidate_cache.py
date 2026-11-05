"""Versioned, complete-only raw detection cache, independent of target association."""

import gzip
import json
from pathlib import Path

from .detector import Detection


class CandidateWriter:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.path = self.directory / "candidates.jsonl.gz"
        self.partial = self.directory / "candidates.jsonl.gz.part"
        self.metadata = metadata
        self.stream = gzip.open(self.partial, "wt", encoding="utf-8", compresslevel=1)
        self.frames = 0

    def write(self, frame, detections):
        self.stream.write(json.dumps({"frame": frame, "detections": [[*d.xyxy, d.confidence] for d in detections]},
                                     separators=(",", ":"), allow_nan=False) + "\n")
        self.frames += 1

    def finish(self):
        from .pipeline import file_hash
        self.stream.close()
        if "requested_frames" in self.metadata and self.frames != self.metadata["requested_frames"]:
            raise ValueError("Incomplete candidate cache; requested frame count was not reached")
        self.partial.replace(self.path)
        metadata = dict(self.metadata, version=1, frames=self.frames, sha256=file_hash(self.path))
        temporary = self.directory / "candidates_meta.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False))
        temporary.replace(self.directory / "candidates_meta.json")

    def close(self):
        self.stream.close()


class CachedBallDetector:
    def __init__(self, directory, video, start_frame=0, max_frames=None, confidence=None):
        from .pipeline import file_hash
        directory, video = Path(directory), Path(video).resolve()
        metadata = json.loads((directory / "candidates_meta.json").read_text())
        path = directory / "candidates.jsonl.gz"
        stat = video.stat()
        if metadata.get("version") != 1 or metadata["video"] != str(video) or metadata["video_size"] != stat.st_size or metadata["video_mtime_ns"] != stat.st_mtime_ns:
            raise ValueError("Candidate cache does not match this source video")
        if file_hash(path) != metadata["sha256"]:
            raise ValueError("Candidate cache checksum mismatch")
        first, count = metadata["start_frame"], metadata["frames"]
        if not first <= start_frame < first + count:
            raise ValueError("Requested start frame is outside the candidate cache")
        available = first + count - start_frame
        if max_frames is not None and max_frames > available:
            raise ValueError("Requested frame count exceeds the candidate cache")
        self.metadata, self.frames = metadata, available
        source_confidence = metadata.get("provenance", {}).get("confidence", 0.)
        if confidence is not None and confidence < source_confidence:
            raise ValueError("A cache cannot recover detections below its original confidence threshold")
        self.confidence = source_confidence if confidence is None else confidence
        self.class_id = metadata.get("class_id")
        self.stream = gzip.open(path, "rt", encoding="utf-8")
        self.next_frame = first
        try:
            for _ in range(start_frame - first):
                self._read()
        except BaseException:
            self.stream.close()
            raise

    def _read(self):
        line = self.stream.readline()
        if not line:
            raise ValueError("Candidate cache ended unexpectedly")
        entry = json.loads(line)
        if entry["frame"] != self.next_frame:
            raise ValueError("Candidate cache frame sequence mismatch")
        self.next_frame += 1
        return [Detection(tuple(row[:4]), float(row[4])) for row in entry["detections"] if row[4] >= self.confidence]

    def detect(self, frame):
        return self._read()

    def close(self):
        self.stream.close()
