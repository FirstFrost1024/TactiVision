"""Streaming export: exactly one row per decoded video frame, including misses."""

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .render import Preview
from .tracking import BallAssociator, Candidate


FIELDS = ["frame", "timestamp_ms", "status", "track_id", "candidate_count", "confidence",
          "u_px", "v_px", "undistorted_u_px", "undistorted_v_px", "x_m", "y_m",
          "x_center_m", "y_center_m", "in_pitch", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct(video, calibration, detector, output, *, space="distorted", start_frame=0,
                max_frames=None, preview=False, max_speed=45., reset_after=.4,
                gate_slack=2., calibration_size=None, provenance=None):
    if space not in ("distorted", "undistorted"):
        raise ValueError("Invalid image space")
    if start_frame < 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("start_frame must be nonnegative and max_frames positive")
    video, output = Path(video), Path(output)
    if not video.is_file():
        raise FileNotFoundError(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {video}")
    writer = None
    counts = {"frames": 0, "detected": 0, "missing": 0, "rejected": 0}
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width, height = (int(capture.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise ValueError("Video has invalid FPS or dimensions")
        if calibration_size and tuple(calibration_size) != (width, height):
            raise ValueError(f"Video {width}x{height} does not match calibration {tuple(calibration_size)}; "
                             "use the original geometry rather than resized/cropped footage")
        if start_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise ValueError("Cannot seek to start frame")
        output.mkdir(parents=True, exist_ok=True)
        associator = BallAssociator(max_speed=max_speed, gate_slack=gate_slack, reset_after=reset_after)
        renderer = Preview(calibration, fps)
        with (output / "ball_positions.csv").open("w", newline="") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=FIELDS)
            csv_writer.writeheader()
            while max_frames is None or counts["frames"] < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index = start_frame + counts["frames"]
                timestamp = frame_index / fps
                detections = detector.detect(frame)
                centers = np.asarray([d.center for d in detections], dtype=float).reshape(-1, 2)
                projected = calibration.image_to_ground(centers, space)
                candidates = [Candidate(d, xy) for d, xy in zip(detections, projected)]
                selected = associator.select(candidates, timestamp)
                status = "detected" if selected else ("rejected" if detections else "missing")
                row = dict.fromkeys(FIELDS, "")
                row.update(frame=frame_index, timestamp_ms=round(timestamp * 1000, 6),
                           status=status, candidate_count=len(detections))
                if selected:
                    u, v = selected.detection.center
                    und = calibration.undistort([[u, v]])[0] if space == "distorted" else (u, v)
                    x, y = selected.pitch_xy
                    cx, cy = calibration.centered([selected.pitch_xy])[0]
                    row.update(track_id=associator.track_id, confidence=selected.detection.confidence,
                               u_px=u, v_px=v, undistorted_u_px=float(und[0]), undistorted_v_px=float(und[1]),
                               x_m=float(x), y_m=float(y), x_center_m=float(cx), y_center_m=float(cy),
                               in_pitch=bool(0 <= x <= calibration.pitch_length and 0 <= y <= calibration.pitch_width))
                    row.update(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), selected.detection.xyxy))
                csv_writer.writerow(row)
                counts["frames"] += 1
                counts[status] += 1
                if preview:
                    rendered = renderer.draw(frame, row)
                    if writer is None:
                        writer = cv2.VideoWriter(str(output / "reconstruction.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                 (rendered.shape[1], rendered.shape[0]))
                        if not writer.isOpened():
                            raise ValueError("Cannot create preview video (mp4v codec required)")
                    writer.write(rendered)
                if counts["frames"] % 100 == 0:
                    print(f"Processed {counts['frames']} frames, {counts['detected']} detections", flush=True)
        if counts["frames"] == 0:
            raise ValueError("No video frames decoded at the requested start frame")
        summary = {
            "video": str(video.resolve()), "fps": fps, "image_size": [width, height],
            "start_frame": start_frame, "image_space": space, **counts,
            "detection_coverage": counts["detected"] / counts["frames"],
            "coordinates": {"unit": "metre", "x_m_y_m": "Original calibration corner origin; x right, y toward near sideline",
                            "x_center_m_y_center_m": "x_m - length/2, width/2 - y_m; x right, y toward far sideline",
                            "pitch_size": [calibration.pitch_length, calibration.pitch_width]},
            "assumptions": ["Ball bounding-box centre is projected onto z=0; airborne-ball height is not recovered.",
                            "Missing/rejected frames have empty coordinates; no interpolation or clamping.",
                            "Frame indices and timestamps refer to this input video, not an inferred full-match timeline.",
                            "Detection coverage is not detection accuracy."],
            "association": {"max_speed_m_s": max_speed, "slack_m": gate_slack, "reset_after_s": reset_after},
            "provenance": provenance or {},
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        return summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()
