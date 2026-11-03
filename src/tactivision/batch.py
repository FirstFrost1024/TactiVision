"""Spawn one persistent model per GPU, dynamically scheduling full half-match videos."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import signal
import tempfile
import time
import traceback

import cv2

from .calibration import Calibration
from .pipeline import file_hash, reconstruct


BATCH_VERSION = 1


@dataclass(frozen=True)
class Job:
    match: str
    half: str
    video: str
    calibration_dir: str
    output: str
    frames: int
    fps: float
    image_size: tuple[int, int]

    @property
    def name(self):
        return f"{self.match}/{self.half}"


def stamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def stop_on_termination():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def write_json(path, value):
    """Publish a fully written status/marker; interrupted writes leave the old file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def discover_jobs(dataset, output, matches=None):
    from .cli import calibration_directory
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    # Use the same recognised layouts as the single-video CLI.
    roots = (dataset, dataset / "data", dataset / "SoccerTrack-v2" / "data",
             dataset / "soccertrack-v2" / "data", dataset / "soccertrack-v2", dataset / "SoccerTrack-v2")
    root = next((p for p in roots if (p / "videos").is_dir()), None)
    if root is None:
        raise FileNotFoundError(f"Cannot find videos/ under {dataset}")
    names = list(dict.fromkeys(matches)) if matches else sorted(p.name for p in (root / "videos").iterdir() if p.is_dir())
    if not names:
        raise ValueError("No matches found")
    jobs = []
    for match in names:
        if not match.isdigit():
            raise ValueError(f"Expected a numeric SoccerTrack-v2 match id: {match}")
        directory = calibration_directory(root, match)
        for suffix in ("camera_intrinsics.npz", "homography.npy", "keypoints.json"):
            path = directory / f"{match}_{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
        for half in ("1st", "2nd"):
            video = root / "videos" / match / f"{match}_panorama_{half}_half.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            capture = cv2.VideoCapture(str(video))
            try:
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = capture.get(cv2.CAP_PROP_FPS)
                size = tuple(int(capture.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
                if not capture.isOpened() or frames <= 0 or not 0 < fps < float("inf") or min(size) <= 0:
                    raise ValueError(f"Invalid video metadata: {video}")
            finally:
                capture.release()
            jobs.append(Job(match, half, str(video), str(directory), str(output / match / half), frames, fps, size))
    return jobs


def fingerprint(job, config):
    stat = Path(job.video).stat()
    files = {p.name: file_hash(p) for p in (
        Path(job.calibration_dir) / f"{job.match}_{suffix}"
        for suffix in ("camera_intrinsics.npz", "homography.npy", "keypoints.json"))}
    settings = {k: v for k, v in config.items() if k not in ("cpu_threads",)}
    value = {"version": BATCH_VERSION, "video": job.video, "video_size": stat.st_size,
             "video_mtime_ns": stat.st_mtime_ns, "frames": job.frames, "calibration": files, "settings": settings,
             "implementation": {name: file_hash(Path(__file__).parent / name) for name in (
                 "batch.py", "pipeline.py", "calibration.py", "detector.py", "tracking.py", "render.py")}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def is_complete(job, signature, preview=False):
    output = Path(job.output)
    try:
        marker = json.loads((output / "completed.json").read_text())
        if marker["signature"] != signature:
            return False
        required = ["ball_positions.csv", "summary.json", "calibration_report.json"]
        if preview:
            required.append("reconstruction.mp4")
        return all((output / name).is_file() and (output / name).stat().st_size > 0
                   and file_hash(output / name) == marker["sha256"][name] for name in required)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def process_job(job, detector, config, signature, device):
    output = Path(job.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "completed.json").unlink(missing_ok=True)
    calibration = Calibration.load(job.calibration_dir, job.match,
                                   pitch_length=config["pitch_length"], pitch_width=config["pitch_width"])
    report = calibration.check_keypoints(Path(job.calibration_dir) / f"{job.match}_keypoints.json")
    report.update(match=job.match, calibration_directory=job.calibration_dir)
    write_json(output / "calibration_report.json", report)
    print(f"Keypoint inverse-projection residual p90={report['pitch_error_m']['p90']:.3f} m "
          "(not a standalone calibration-validity test or ball localisation accuracy)", flush=True)
    provenance = {"match": job.match, "half": job.half, "dataset": "SoccerTrack-v2", "worker_device": device,
                  "weights": config["weights"], "weights_sha256": config["weights_sha256"],
                  "class_id": detector.class_id, "batch_signature": signature,
                  **{key: config[key] for key in ("confidence", "imgsz", "tile_size", "overlap")}}
    summary = reconstruct(job.video, calibration, detector, output,
                          **{key: config[key] for key in ("space", "max_frames", "preview", "max_speed", "gate_slack", "reset_after")},
                          calibration_size=job.image_size, provenance=provenance)
    expected = min(job.frames, config["max_frames"]) if config["max_frames"] else job.frames
    if summary["frames"] != expected:
        raise ValueError(f"Incomplete decode: expected {expected} frames, received {summary['frames']}; not marked complete")
    files = ["ball_positions.csv", "summary.json", "calibration_report.json"]
    if config["preview"]:
        files.append("reconstruction.mp4")
    write_json(output / "completed.json", {"signature": signature, "finished_at": stamp(),
               "frames": summary["frames"], "sha256": {name: file_hash(output / name) for name in files}})
    return {"frames": summary["frames"], "detected": summary["detected"]}


def worker(device, visible_device, config, jobs, events):
    """CUDA is initialised only inside a fresh child with exactly one visible GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
    os.environ["OMP_NUM_THREADS"] = str(config["cpu_threads"])
    detector = None
    try:
        # Import torch only after masking devices. Every child uses its own local cuda:0.
        import torch
        from .detector import BallDetector
        torch.set_num_threads(config["cpu_threads"])
        cv2.setNumThreads(1)
        detector = BallDetector(**{key: config[key] for key in (
            "weights", "confidence", "imgsz", "tile_size", "overlap", "class_id")}, device=torch.device("cuda:0"))
        detector.model.to("cuda:0")
        events.put({"type": "ready", "device": device, "pid": os.getpid()})
        while True:
            item = jobs.get()
            if item is None:
                return
            job, signature = item
            output = Path(job.output)
            output.mkdir(parents=True, exist_ok=True)
            events.put({"type": "started", "job": job.name, "device": device})
            started = time.monotonic()
            with (output / "processing.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
                print(f"{stamp()} Start {job.name}, GPU {device}, pid={os.getpid()}", flush=True)
                try:
                    result = process_job(job, detector, config, signature, device)
                except Exception as error:
                    traceback.print_exc()
                    events.put({"type": "failed", "job": job.name, "device": device, "error": str(error)})
                    # Release unused cached tensors before trying another video after an OOM.
                    torch.cuda.empty_cache()
                else:
                    events.put({"type": "completed", "job": job.name, "device": device,
                                "elapsed_s": round(time.monotonic() - started, 2), **result})
    except Exception:
        events.put({"type": "fatal", "device": device, "error": traceback.format_exc()})


def validate_devices(devices):
    import torch
    if not devices or len(set(devices)) != len(devices) or any(not d.isdigit() for d in devices):
        raise ValueError("--devices requires distinct GPU indices, e.g. --devices 0 1")
    count = torch.cuda.device_count()
    if any(int(d) >= count for d in devices):
        raise ValueError(f"Requested GPUs {devices}, but only {count} CUDA devices are visible")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = visible.split(",") if visible else [str(i) for i in range(count)]
    return [tokens[int(d)].strip() for d in devices]


def schedule(pending, devices, visible_devices, config, status, status_path, target=worker):
    context = mp.get_context("spawn")
    tasks, events = context.Queue(), context.Queue()
    processes = []
    finished = 0
    try:
        for item in pending:
            tasks.put(item)
        for _ in devices:
            tasks.put(None)
        for device, visible in zip(devices, visible_devices):
            process = context.Process(target=target, args=(device, visible, config, tasks, events), name=f"tactivision-gpu-{device}")
            process.start()
            processes.append(process)
        while finished < len(pending):
            try:
                event = events.get(timeout=1)
            except Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or not any(p.is_alive() for p in processes):
                    raise RuntimeError("A GPU worker exited before all video results were returned; rerun to resume")
                continue
            kind = event["type"]
            if kind == "fatal":
                raise RuntimeError(f"GPU {event['device']} worker failed: {event['error']}")
            if kind == "ready":
                print(f"GPU {event['device']} model ready (pid={event['pid']})", flush=True)
                continue
            state = status["jobs"][event["job"]]
            state.update(event, updated_at=stamp(), status=kind)
            write_json(status_path, status)
            print(f"[{kind}] GPU {event['device']} {event['job']} "
                  f"{event.get('frames', '')} {event.get('error', '')}", flush=True)
            if kind in ("completed", "failed"):
                finished += 1
        return sum(item["status"] == "failed" for item in status["jobs"].values())
    finally:
        for process in processes:
            if finished < len(pending) and process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        tasks.cancel_join_thread()
        tasks.close()
        events.close()


def run_batch(args):
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    jobs = discover_jobs(args.dataset, args.output, args.matches)
    total = sum(min(job.frames, args.max_frames) if args.max_frames else job.frames for job in jobs)
    print(f"发现 {len({j.match for j in jobs})} 场比赛、{len(jobs)} 个半场视频，共 {total:,} 待处理帧", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"{job.name}: {job.frames:,} frames, {job.image_size[0]}x{job.image_size[1]}, {job.fps:g} fps -> {job.output}")
        return {"planned": len(jobs), "frames": total}
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Local detector weights missing: {weights}")
    visible = validate_devices(args.devices)
    keys = ("confidence", "class_id", "imgsz", "tile_size", "overlap", "space", "max_frames", "preview",
            "max_speed", "gate_slack", "reset_after", "pitch_length", "pitch_width", "cpu_threads")
    config = {key: getattr(args, key) for key in keys}
    config.update(weights=str(weights), weights_sha256=file_hash(weights))
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "batch.lock").open("a+") as lock, stop_on_termination():
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another batch is using {root}; choose another output or wait") from error
        status = {"started_at": stamp(), "pid": os.getpid(), "devices": args.devices,
                  "config": config, "jobs": {}}
        pending = []
        for job in jobs:
            signature = fingerprint(job, config)
            complete = is_complete(job, signature, args.preview)
            status["jobs"][job.name] = {"status": "skipped" if complete else "pending", **asdict(job)}
            if not complete:
                pending.append((job, signature))
        path = root / "batch_status.json"
        write_json(path, status)
        print(f"跳过已完成 {len(jobs) - len(pending)} 段，排队 {len(pending)} 段；GPU {', '.join(args.devices)}", flush=True)
        try:
            failed = schedule(pending, args.devices, visible, config, status, path) if pending else 0
        except BaseException:
            status.update(interrupted_at=stamp(), interrupted=True)
            write_json(path, status)
            raise
        status.update(finished_at=stamp(), failed=failed)
        write_json(path, status)
        print(f"批处理结束：完成 {len(pending) - failed} 段，跳过 {len(jobs) - len(pending)} 段，失败 {failed} 段", flush=True)
        return {"completed": len(pending) - failed, "skipped": len(jobs) - len(pending), "failed": failed}
