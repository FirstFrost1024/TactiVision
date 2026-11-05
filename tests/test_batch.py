import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Queue
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import tactivision.batch as batch
from tactivision.cli import main, parser
from tactivision.pipeline import file_hash


def dataset_fixture(tmp_path, match="117093"):
    root = tmp_path / "dataset"
    raw = root / "raw" / match
    raw.mkdir(parents=True)
    K = np.array([[300., 0, 320], [0, 310., 240], [0, 0, 1]])
    H = np.array([[4., 0, 100], [0, 3., 100], [0, 0, 1.]])
    from tactivision.calibration import Calibration
    calibration = Calibration(K, np.zeros(4), K, H)
    np.savez(raw / f"{match}_camera_intrinsics.npz", K=K, D=np.zeros(4), Knew=K)
    np.save(raw / f"{match}_homography.npy", H)
    world = [[0, 0], [105, 0], [0, 68], [105, 68]]
    (raw / f"{match}_keypoints.json").write_text(json.dumps({
        f"({x},{y})": p.tolist() for (x, y), p in zip(world, calibration.ground_to_image(world))}))
    for half in ("1st", "2nd"):
        video = root / "videos" / match / f"{match}_panorama_{half}_half.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25., (640, 480))
        assert writer.isOpened()
        for _ in range(3):
            writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        writer.release()
    return root


def config_fixture(tmp_path):
    weights = tmp_path / "local.pt"
    weights.write_bytes(b"fake weights for controlled test")
    return dict(weights=str(weights), weights_sha256=file_hash(weights), confidence=.15,
                class_id=None, imgsz=1024, tile_size=1024, overlap=.2, space="distorted",
                max_frames=None, preview=False, max_speed=45., gate_slack=2., reset_after=.4,
                pitch_length=105., pitch_width=68., cpu_threads=1)


def test_discover_both_halves_and_fail_on_missing_video(tmp_path):
    dataset = dataset_fixture(tmp_path)
    jobs = batch.discover_jobs(dataset, tmp_path / "out")
    assert [j.name for j in jobs] == ["117093/1st", "117093/2nd"]
    assert all(j.frames == 3 and j.fps == 25. for j in jobs)
    Path(jobs[1].video).unlink()
    with pytest.raises(FileNotFoundError):
        batch.discover_jobs(dataset, tmp_path / "out")


def test_completed_outputs_skip_only_matching_inputs_and_untampered_artifacts(tmp_path):
    dataset = dataset_fixture(tmp_path)
    job = batch.discover_jobs(dataset, tmp_path / "out")[0]
    config = config_fixture(tmp_path)
    signature = batch.fingerprint(job, config)
    detector = SimpleNamespace(detect=lambda frame: [], class_id=32)
    result = batch.process_job(job, detector, config, signature, "0")
    assert result["frames"] == 3
    assert batch.is_complete(job, signature)
    config["confidence"] = .3
    assert not batch.is_complete(job, batch.fingerprint(job, config))
    (Path(job.output) / "ball_positions.csv").write_text("truncated")
    assert not batch.is_complete(job, signature)


def test_short_smoke_cannot_skip_full_video(tmp_path):
    dataset = dataset_fixture(tmp_path)
    job = batch.discover_jobs(dataset, tmp_path / "out")[0]
    config = config_fixture(tmp_path)
    config["max_frames"] = 1
    signature = batch.fingerprint(job, config)
    batch.process_job(job, SimpleNamespace(detect=lambda frame: [], class_id=32), config, signature, "0")
    assert batch.is_complete(job, signature)
    config["max_frames"] = None
    assert not batch.is_complete(job, batch.fingerprint(job, config))


def test_completed_candidate_cache_is_covered_by_batch_integrity(tmp_path):
    dataset=dataset_fixture(tmp_path)
    job=batch.discover_jobs(dataset,tmp_path/'out')[0]
    config=config_fixture(tmp_path)
    signature=batch.fingerprint(job,config)
    batch.process_job(job,SimpleNamespace(detect=lambda frame:[],class_id=32),config,signature,'0')
    assert batch.is_complete(job,signature)
    cache=Path(job.output)/'candidates.jsonl.gz'
    assert cache.exists()
    with cache.open('ab') as stream:stream.write(b'changed')
    assert not batch.is_complete(job,signature)


def test_truncated_decode_not_marked_complete(tmp_path, monkeypatch):
    dataset = dataset_fixture(tmp_path)
    job = batch.discover_jobs(dataset, tmp_path / "out")[0]
    config = config_fixture(tmp_path)
    monkeypatch.setattr(batch, "reconstruct", lambda *a, **kw: {"frames": 1})
    with pytest.raises(ValueError, match="Incomplete decode"):
        batch.process_job(job, SimpleNamespace(class_id=32), config, "signature", "0")
    assert not (Path(job.output) / "completed.json").exists()


def test_worker_reuses_model_and_continues_after_video_failure(tmp_path, monkeypatch):
    config = config_fixture(tmp_path)
    tasks, events = Queue(), Queue()
    jobs = [batch.Job("117093", half, "unused", "unused", str(tmp_path / half), 3, 25., (640, 480))
            for half in ("1st", "2nd")]
    for job in jobs:
        tasks.put((job, "signature"))
    tasks.put(None)
    models, handled = [], []
    def detector_factory(**kwargs):
        models.append(kwargs)
        return SimpleNamespace(model=SimpleNamespace(to=lambda device: None))
    def process(job, detector, *args):
        handled.append(job.name)
        if job.half == "1st":
            raise ValueError("controlled failure")
        return {"frames": 3, "detected": 0}
    import tactivision.detector
    monkeypatch.setattr(tactivision.detector, "BallDetector", detector_factory)
    monkeypatch.setattr(batch, "process_job", process)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "initial")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    fake_torch = SimpleNamespace(set_num_threads=lambda count: None, device=lambda name: name,
                                 cuda=SimpleNamespace(empty_cache=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    batch.worker("1", "GPU-example", config, tasks, events)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-example"
    assert len(models) == 1 and handled == [job.name for job in jobs]
    assert [events.get()["type"] for _ in range(events.qsize())] == [
        "ready", "started", "failed", "started", "completed"]


def fake_spawn_worker(device, visible, config, tasks, events):
    config["barrier"].wait(timeout=15)
    while True:
        item = tasks.get()
        if item is None:
            return
        job, signature = item
        time.sleep(.05)
        events.put({"type": "completed", "job": job.name, "device": device,
                    "pid": os.getpid(), "frames": job.frames})


def test_spawn_scheduler_processes_every_job_with_two_independent_workers(tmp_path):
    jobs = [batch.Job(str(100000 + i), "1st", "unused", "unused", "unused", 3, 25., (640, 480)) for i in range(4)]
    status = {"jobs": {j.name: {"status": "pending"} for j in jobs}}
    config = {"barrier": mp.get_context("spawn").Barrier(2)}
    failed = batch.schedule([(j, "sig") for j in jobs], ["0", "1"], ["0", "1"], config,
                            status, tmp_path / "status.json", target=fake_spawn_worker)
    assert failed == 0
    states = status["jobs"].values()
    assert all(s["status"] == "completed" for s in states)
    assert len({s["pid"] for s in states}) == 2
    assert {s["device"] for s in states} == {"0", "1"}


def test_dry_run_uses_full_dataset_without_loading_weights_or_writing_outputs(tmp_path, capsys):
    root = dataset_fixture(tmp_path)
    output = tmp_path / "out"
    main(["batch", "--dataset", str(root), "--weights", "not-downloaded.pt",
          "--output", str(output), "--dry-run"])
    assert "2 个半场视频" in capsys.readouterr().out
    assert not output.exists()


def test_gpu_indices_respect_parent_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa,GPU-bbb")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)))
    assert batch.validate_devices(["1", "0"]) == ["GPU-bbb", "GPU-aaa"]
    with pytest.raises(ValueError):
        batch.validate_devices(["0", "0"])
    with pytest.raises(ValueError):
        batch.validate_devices(["2"])
