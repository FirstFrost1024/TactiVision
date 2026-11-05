"""Local video preview of image detections and ground-plane projections."""

from collections import deque

import cv2
import numpy as np


class Preview:
    def __init__(self, calibration, fps, region=None):
        self.calibration = calibration
        self.trail = deque(maxlen=max(1, round(fps * 2)))
        self.previous_track = None
        self.region = region

    def draw(self, frame, row):
        width = 1280
        video_height = round(frame.shape[0] * width / frame.shape[1] / 2) * 2
        canvas = np.full((video_height + 480, width, 3), (27, 24, 20), dtype=np.uint8)
        image = frame.copy()
        if self.region is not None:
            cv2.polylines(image, [np.rint(self.region.vertices).astype(np.int32)], True, (110, 200, 130), 1, cv2.LINE_AA)
        detected = row["status"] == "detected"
        if detected:
            box = [int(round(row[k])) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
            cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 220, 255), 3)
            cv2.circle(image, (round(row["u_px"]), round(row["v_px"])), 5, (0, 220, 255), -1)
        canvas[:video_height] = cv2.resize(image, (width, video_height))
        origin = np.array([40, video_height + 45])
        sx, sy = 760 / self.calibration.pitch_length, 390 / self.calibration.pitch_width

        def point(x, y):
            return tuple(np.rint(origin + (x * sx, y * sy)).astype(int))

        def line(a, b):
            cv2.line(canvas, point(*a), point(*b), (195, 214, 202), 2, cv2.LINE_AA)

        L, W = self.calibration.pitch_length, self.calibration.pitch_width
        cv2.rectangle(canvas, point(0, 0), point(L, W), (45, 92, 54), -1)
        cv2.rectangle(canvas, point(0, 0), point(L, W), (195, 214, 202), 2)
        line((L / 2, 0), (L / 2, W))
        cv2.ellipse(canvas, point(L / 2, W / 2), (round(9.15 * sx), round(9.15 * sy)),
                    0, 0, 360, (195, 214, 202), 2, cv2.LINE_AA)
        for left, right in ((0, 16.5), (L - 16.5, L)):
            cv2.rectangle(canvas, point(left, W / 2 - 20.16), point(right, W / 2 + 20.16),
                          (195, 214, 202), 2)
        if not detected or row["track_id"] != self.previous_track:
            self.trail.clear()
        if detected:
            self.previous_track = row["track_id"]
            if row["in_pitch"]:
                self.trail.append(point(row["x_m"], row["y_m"]))
                if len(self.trail) > 1:
                    cv2.polylines(canvas, [np.array(self.trail)], False, (0, 190, 235), 2, cv2.LINE_AA)
                cv2.circle(canvas, self.trail[-1], 7, (0, 220, 255), -1, cv2.LINE_AA)
        texts = ["TACTIVISION / BALL", f"Frame {row['frame']} | {row['timestamp_ms'] / 1000:.2f}s",
                 f"Status: {row['status']}", "Ground-plane assumption: z=0"]
        if detected:
            texts += [f"Pitch: ({row['x_m']:.2f}, {row['y_m']:.2f}) m",
                      f"Confidence: {row['confidence']:.3f}",
                      "Inside pitch" if row["in_pitch"] else "Outside pitch (position retained)"]
            # A crop exposes small-ball mistakes that are invisible in a panorama.
            u, v = round(row["u_px"]), round(row["v_px"])
            crop = frame[max(0, v - 45):min(frame.shape[0], v + 45),
                         max(0, u - 70):min(frame.shape[1], u + 70)]
            if crop.size:
                canvas[video_height + 260:video_height + 440, 865:1145] = cv2.resize(crop, (280, 180))
        for i, text in enumerate(texts):
            cv2.putText(canvas, text, (845, video_height + 40 + 29 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (225, 235, 238), 1, cv2.LINE_AA)
        return canvas
