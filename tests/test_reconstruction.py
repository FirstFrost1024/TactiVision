import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tactivision.calibration import Calibration, transform
from tactivision.cli import calibration_directory, main
from tactivision.detector import BallDetector, Detection, suppress_duplicates, tile_starts
from tactivision.pipeline import reconstruct
from tactivision.tracking import BallAssociator, Candidate


@pytest.fixture
def calibration():
    K = np.array([[300., 0, 320], [0, 310., 240], [0, 0, 1]])
    D = np.array([-.05, .01, 0., 0.])
    Knew = np.array([[250., 0, 320], [0, 260., 240], [0, 0, 1]])
    H = np.array([[4., -.2, 100], [.1, 3., 100], [.001, .002, 1]])
    return Calibration(K, D, Knew, H)


def save_calibration(directory, calibration, match="sample"):
    directory.mkdir(parents=True, exist_ok=True)
    # Match upstream's NPZ schema, including object arrays we must not unpickle.
    np.savez(directory / f"{match}_camera_intrinsics.npz", K=calibration.K,
             D=calibration.D, Knew=calibration.Knew, rvecs=np.array([None], dtype=object))
    np.save(directory / f"{match}_homography.npy", calibration.pitch_to_image)
    world = np.array([[0, 0], [105, 0], [0, 68], [105, 68], [52.5, 34]])
    pixels = calibration.ground_to_image(world)
    (directory / f"{match}_keypoints.json").write_text(json.dumps(
        {f"({x},{y})": p.tolist() for (x, y), p in zip(world, pixels)}))


@pytest.mark.parametrize("space", ["distorted", "undistorted"])
def test_projection_roundtrip_and_homography_direction(calibration, space):
    world = np.array([[0, 0], [52.5, 34], [105, 68], [-3, 70]], dtype=float)
    pixels = calibration.ground_to_image(world, space)
    np.testing.assert_allclose(calibration.image_to_ground(pixels, space), world, atol=1e-6)
    if space == "undistorted":
        np.testing.assert_allclose(pixels, transform(world, calibration.pitch_to_image))


def test_empty_projection_and_horizon(calibration):
    assert calibration.image_to_ground([]).shape == (0, 2)
    H = np.array([[1., 0, 0], [0, 1, 0], [1, 0, -1]])
    assert np.isnan(transform([[1, 3]], H)).all()


def test_center_origin_and_y_axis(calibration):
    np.testing.assert_allclose(calibration.centered([[0, 0], [105, 68], [52.5, 34]]),
                               [[-52.5, 34], [52.5, -34], [0, 0]])


def test_numeric_only_upstream_npz_load_and_keypoint_report(tmp_path, calibration):
    save_calibration(tmp_path, calibration)
    loaded = Calibration.load(tmp_path, "sample")
    report = loaded.check_keypoints(tmp_path / "sample_keypoints.json")
    assert report["keypoint_count"] == 5
    assert report["pitch_error_m"]["max"] < 1e-6


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan)])
def test_bad_homography_rejected(calibration, bad):
    with pytest.raises(ValueError):
        Calibration(calibration.K, calibration.D, calibration.Knew, bad)


def test_tiles_cover_edges_and_small_inputs():
    assert tile_starts(100, 1024, .2) == [0]
    starts = tile_starts(4096, 1024, .2)
    assert starts[0] == 0 and starts[-1] + 1024 == 4096
    assert all(b - a <= 1024 for a, b in zip(starts, starts[1:]))
    with pytest.raises(ValueError):
        tile_starts(100, 30, 1)


def test_nms_suppresses_overlapping_tile_duplicates():
    ds = [Detection((10, 20, 20, 30), .6), Detection((11, 20, 21, 30), .8),
          Detection((100, 200, 110, 210), .4)]
    kept = suppress_duplicates(ds)
    assert len(kept) == 2
    assert ds[1] in kept and ds[2] in kept


def test_yolo_tiles_restore_full_resolution_coordinates(monkeypatch, tmp_path):
    class Tensor:
        def __init__(self, value): self.value = np.array(value)
        def cpu(self): return self
        def numpy(self): return self.value

    class Model:
        names = {0: "ball"}
        def __init__(self, weights): self.calls = []
        def predict(self, crop, **kwargs):
            self.calls.append((crop.shape, kwargs))
            boxes = SimpleNamespace(xyxy=Tensor([[20, 30, 30, 40]]), conf=Tensor([.9]))
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    weights = tmp_path / "local.pt"
    weights.touch()
    detector = BallDetector(weights, tile_size=100, overlap=0)
    result = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert len(result) == 2
    assert {d.center for d in result} == {(25, 35), (125, 35)}
    assert detector.model.calls[0][1]["classes"] == [0]


def candidate(x, y, confidence=.8):
    return Candidate(Detection((0, 0, 4, 4), confidence), np.array([x, y], dtype=float))


def test_association_rejects_jump_and_reacquires_after_gap():
    tracker = BallAssociator(max_speed=40, gate_slack=1, reset_after=.4)
    assert tracker.select([candidate(0, 0)], 0) is not None
    selected = tracker.select([candidate(1, 0, .5), candidate(90, 0, .99)], .04)
    assert selected.pitch_xy[0] == 1
    assert tracker.select([candidate(90, 0)], .08) is None
    assert tracker.select([], .12) is None
    assert tracker.select([candidate(90, 0)], .6) is not None
    assert tracker.track_id == 2


def test_nonfinite_projection_not_selected():
    assert BallAssociator().select([candidate(np.nan, 2)], 0) is None


def write_video(path, frames=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25., (640, 480))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((480, 640, 3), i * 20, dtype=np.uint8))
    writer.release()


def test_video_pipeline_frame_alignment_misses_rejection_and_preview(tmp_path, calibration):
    video = tmp_path / "input.mp4"
    write_video(video)

    class Detector:
        def __init__(self): self.i = 0
        def detect(self, frame):
            # A visible frame, a miss, an impossible jump, then a nearby detection.
            i = self.i
            self.i += 1
            if i == 1: return []
            world = [[100, 65]] if i == 2 else [[10 + i * .1, 20]]
            u, v = calibration.ground_to_image(world)[0]
            return [Detection((u - 2, v - 2, u + 2, v + 2), .9)]

    out = tmp_path / "result"
    summary = reconstruct(video, calibration, Detector(), out, start_frame=1,
                          max_frames=4, preview=True, calibration_size=(640, 480))
    rows = list(csv.DictReader((out / "ball_positions.csv").open()))
    assert [int(r["frame"]) for r in rows] == [1, 2, 3, 4]
    assert [float(r["timestamp_ms"]) for r in rows] == [40, 80, 120, 160]
    assert [r["status"] for r in rows] == ["detected", "missing", "rejected", "detected"]
    assert rows[1]["x_m"] == rows[2]["confidence"] == ""
    assert float(rows[0]["x_m"]) == pytest.approx(10, abs=1e-6)
    assert summary["detected"] == 2 and summary["detection_coverage"] == .5
    result = cv2.VideoCapture(str(out / "reconstruction.mp4"))
    assert result.get(cv2.CAP_PROP_FRAME_COUNT) == 4
    assert result.read()[0]
    result.release()


def test_video_geometry_mismatch_fails(tmp_path, calibration):
    video = tmp_path / "input.mp4"
    write_video(video)
    with pytest.raises(ValueError, match="does not match"):
        reconstruct(video, calibration, None, tmp_path / "out", calibration_size=(4096, 1080))


def test_outside_pitch_coordinate_is_not_clamped(tmp_path, calibration):
    video = tmp_path / "input.mp4"
    write_video(video, frames=1)
    u, v = calibration.ground_to_image([[-2., 34.]])[0]
    detector = SimpleNamespace(detect=lambda frame: [Detection((u - 1, v - 1, u + 1, v + 1), .9)])
    out = tmp_path / "out"
    reconstruct(video, calibration, detector, out)
    row = next(csv.DictReader((out / "ball_positions.csv").open()))
    assert float(row["x_m"]) == pytest.approx(-2., abs=1e-6)
    assert row["in_pitch"] == "False"


def test_cli_local_dataset_layout_and_report(tmp_path, calibration):
    raw = tmp_path / "raw" / "sample"
    save_calibration(raw, calibration)
    assert calibration_directory(tmp_path, "sample") == raw
    out = tmp_path / "output"
    main(["check-calibration", "--dataset", str(tmp_path), "--match", "sample", "--output", str(out)])
    report = json.loads((out / "calibration_report.json").read_text())
    assert len(report["sha256"]) == 3


def test_cli_missing_local_weights_does_not_invoke_yolo(tmp_path, calibration, monkeypatch):
    raw = tmp_path / "raw" / "sample"
    save_calibration(raw, calibration)
    def forbidden(*args, **kwargs):
        pytest.fail("Missing local weights must not invoke YOLO or network downloads")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=forbidden))
    with pytest.raises(SystemExit) as exc:
        main(["run", "--dataset", str(tmp_path), "--match", "sample", "--video", "unused.mp4",
              "--weights", str(tmp_path / "absent.pt"), "--output", str(tmp_path / "out")])
    assert exc.value.code == 2


def test_detector_api_requires_local_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="No automatic download"):
        BallDetector(tmp_path / "nonexistent.pt")


def test_nested_upstream_repository_dataset_layout(tmp_path, calibration):
    raw = tmp_path / "SoccerTrack-v2" / "data" / "raw" / "sample"
    save_calibration(raw, calibration)
    assert calibration_directory(tmp_path, "sample") == raw


def test_cli_default_dataset_and_output_independent_of_working_directory(tmp_path, calibration, monkeypatch):
    import tactivision.cli as cli
    project = tmp_path / "project"
    dataset = project / "datasets" / "SoccerTrack-v2" / "data"
    save_calibration(dataset / "raw" / "sample", calibration)
    monkeypatch.setattr(cli, "DEFAULT_DATASET", dataset)
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT", project / "outputs")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    main(["check-calibration", "--match", "sample"])
    report = json.loads((project / "outputs" / "calibration_report.json").read_text())
    assert Path(report["calibration_directory"]) == dataset / "raw" / "sample"
    assert not (elsewhere / "outputs").exists()


def test_large_residual_preserved_without_declaring_official_calibration_invalid(tmp_path, calibration, capsys):
    raw = tmp_path / "raw" / "sample"
    save_calibration(raw, calibration)
    points = raw / "sample_keypoints.json"
    data = json.loads(points.read_text())
    points.write_text(json.dumps({key: [u + 50, v + 50] for key, (u, v) in data.items()}))
    originals = {path: path.read_bytes() for path in raw.iterdir()}
    out = tmp_path / "out"
    main(["check-calibration", "--dataset", str(tmp_path), "--match", "sample", "--output", str(out)])
    report = json.loads((out / "calibration_report.json").read_text())
    assert report["pitch_error_m"]["p90"] > 2
    assert "do not, by themselves" in report["note"]
    captured = capsys.readouterr()
    assert "不能单独判定" in captured.out
    assert captured.err == ""
    assert all(path.read_bytes() == contents for path, contents in originals.items())
