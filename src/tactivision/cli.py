"""Local-only CLI; dataset files are supplied by the user."""

import argparse
import json
from pathlib import Path
import sys

from .calibration import Calibration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "SoccerTrack-v2" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs"


def calibration_directory(root: Path, match: str) -> Path:
    roots = (root, root / "data", root / "soccertrack-v2", root / "SoccerTrack-v2",
             root / "soccertrack-v2" / "data", root / "SoccerTrack-v2" / "data")
    for directory in (*(base / "raw" / match for base in roots), root / match, root):
        if (directory / f"{match}_camera_intrinsics.npz").is_file():
            return directory
    raise FileNotFoundError(f"No {match}_camera_intrinsics.npz found under {root}. "
                            f"Expected {root / 'raw' / match}/ or supply --calibration-dir.")


def parser():
    p = argparse.ArgumentParser(description="SoccerTrack-v2 足球检测与逐帧球场坐标重建（本地数据）")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("check-calibration", "run", "batch"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                         help=f"数据根目录（默认：{DEFAULT_DATASET}）")
        if name != "batch":
            cmd.add_argument("--match", required=True)
            cmd.add_argument("--calibration-dir", type=Path)
        cmd.add_argument("--pitch-length", type=float, default=105.)
        cmd.add_argument("--pitch-width", type=float, default=68.)
        cmd.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                         help=f"输出目录（默认：{DEFAULT_OUTPUT}）")
    run = sub.choices["run"]
    run.add_argument("--video", required=True, type=Path)
    run.add_argument("--device", default="cpu")
    run.add_argument("--calibration-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    run.add_argument("--start-frame", type=int, default=0)
    run.add_argument("--candidate-cache", type=Path, help="读取该目录的完整候选缓存，跳过 YOLO 推理")
    for cmd in (run, sub.choices["batch"]):
        cmd.add_argument("--weights", required=True, help="本地 YOLO 检测权重路径 (.pt)，不自动下载")
        cmd.add_argument("--confidence", type=float, default=.15)
        cmd.add_argument("--class-id", type=int, help="自定义模型足球类别编号；默认按模型类别名称识别")
        cmd.add_argument("--imgsz", type=int, default=1024)
        cmd.add_argument("--tile-size", type=int, default=1024, help="全景图切片尺寸；0 表示整帧")
        cmd.add_argument("--overlap", type=float, default=.2)
        cmd.add_argument("--space", choices=("distorted", "undistorted"), default="distorted")
        cmd.add_argument("--max-frames", type=int)
        cmd.add_argument("--preview", action="store_true")
        cmd.add_argument("--max-speed", type=float, default=45.)
        cmd.add_argument("--gate-slack", type=float, default=2.)
        cmd.add_argument("--reset-after", type=float, default=.4)
        cmd.add_argument("--pitch-region", choices=("keypoints", "none"), default="keypoints",
                         help="默认从官方原图关键点建立比赛区域；none 复现旧的无区域关联")
        cmd.add_argument("--off-field-grace", type=float, default=1., help="已建立轨迹离开比赛区域后允许延续的秒数")
        cmd.add_argument("--outside-static-seconds", type=float, default=1., help="场外候选静止多久后抑制；0 关闭")
        cmd.add_argument("--save-candidates", action=argparse.BooleanOptionalAction, default=True,
                         help="保存所有原始检测候选，供之后复用；默认开启")
        cmd.add_argument("--focus-pitch", action=argparse.BooleanOptionalAction, default=True,
                         help="在比赛区域的包围框及32px边缘内检测，减少全景空白；默认开启")
    batch = sub.choices["batch"]
    batch.add_argument("--devices", nargs="+", default=["0", "1"], help="每卡一个进程；GPU 编号相对于 CUDA_VISIBLE_DEVICES")
    batch.add_argument("--matches", nargs="+", help="可选比赛编号；默认处理 videos/ 下全部比赛的两个半场")
    batch.add_argument("--cpu-threads", type=int, default=2, help="每个 GPU 进程的 CPU 计算线程数")
    batch.add_argument("--dry-run", action="store_true", help="检查数据并列出任务，不加载模型、不创建输出")
    review = sub.add_parser("review", help="核验已完成的重建结果，并导出轨迹图、画面抽检和短视频")
    review.add_argument("--input", type=Path, default=DEFAULT_OUTPUT / "all_matches")
    review.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "review")
    review.add_argument("--samples", type=int, default=4, help="每半场均匀抽检帧数，另加静止段和最大像素跳变样本")
    review.add_argument("--clip-seconds", type=float, default=4., help="每半场生成的短片秒数；0 表示只导出图像")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "review":
            from .review import run_review
            report = run_review(args.input, args.output, args.samples, args.clip_seconds)
            if report["integrity_passed"] != report["half_count"]:
                p.exit(1, "部分重建文件核验未通过，详情见 quality_report.json。\n")
            return
        if args.command == "batch":
            from .batch import run_batch
            result = run_batch(args)
            if result.get("failed", 0):
                p.exit(1, "部分视频处理失败，详情见 batch_status.json 和各半场 processing.log。\n")
            return
        directory = args.calibration_dir or calibration_directory(args.dataset, args.match)
        calibration = Calibration.load(directory, args.match, pitch_length=args.pitch_length,
                                       pitch_width=args.pitch_width)
        keypoints = directory / f"{args.match}_keypoints.json"
        if not keypoints.is_file():
            raise FileNotFoundError(f"Missing calibration validation keypoints: {keypoints}")
        report = calibration.check_keypoints(keypoints)
        # Do not interpret this as a benchmark: these are the calibration's own keypoints.
        from .pipeline import file_hash, reconstruct
        report.update(match=args.match, calibration_directory=str(directory.resolve()),
                      sha256={f.name: file_hash(f) for f in (
                          directory / f"{args.match}_camera_intrinsics.npz",
                          directory / f"{args.match}_homography.npy", keypoints)})
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "calibration_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        stats = report["pitch_error_m"]
        print(f"Keypoint inverse-projection residuals: median={stats['median']:.3f} m, "
              f"p90={stats['p90']:.3f} m, max={stats['max']:.3f} m")
        print("以上为关键点与投影链的残差，不能单独判定官方标定有效性或足球定位精度。")
        if args.command == "check-calibration":
            return
        if not Path(args.weights).is_file():
            raise FileNotFoundError(f"Local detector weights missing: {args.weights}. No automatic download performed.")
        if args.candidate_cache:
            from .candidate_cache import CachedBallDetector
            detector = CachedBallDetector(args.candidate_cache, args.video, args.start_frame, args.max_frames,
                                          confidence=args.confidence)
            if detector.metadata["image_space"] != args.space:
                detector.close()
                raise ValueError("Candidate cache uses a different image coordinate space")
            source = detector.metadata.get("provenance", {})
            if source.get("weights_sha256") and source["weights_sha256"] != file_hash(args.weights):
                detector.close()
                raise ValueError("Requested weights do not match the model that generated the cache")
            if args.class_id is not None and args.class_id != detector.class_id:
                detector.close()
                raise ValueError("Requested ball class does not match the candidate cache")
            if args.max_frames is None:
                args.max_frames = detector.frames
        else:
            from .detector import BallDetector
            detector = BallDetector(weights=args.weights, confidence=args.confidence, imgsz=args.imgsz,
                                    tile_size=args.tile_size, overlap=args.overlap,
                                    device=args.device, class_id=args.class_id)
        from .region import PitchRegion
        region = PitchRegion.load(keypoints, calibration, args.space) if args.pitch_region == "keypoints" else None
        if not args.candidate_cache:
            detector.focus_region = region if args.focus_pitch else None
        provenance = {"match": args.match, "dataset": "SoccerTrack-v2",
                      "calibration_sha256": report["sha256"], "weights": str(Path(args.weights).resolve()),
                      "weights_sha256": file_hash(args.weights), "class_id": detector.class_id,
                      "confidence": args.confidence, "imgsz": args.imgsz,
                      "tile_size": args.tile_size, "overlap": args.overlap,
                      "focus_pitch": args.focus_pitch and region is not None}
        if args.candidate_cache:
            provenance["candidate_cache"] = str(args.candidate_cache.resolve())
            provenance["source_detection_provenance"] = detector.metadata.get("provenance", {})
            for key in ("imgsz", "tile_size", "overlap", "focus_pitch"):
                if key in detector.metadata.get("provenance", {}):
                    provenance[key] = detector.metadata["provenance"][key]
        try:
            summary = reconstruct(args.video, calibration, detector, args.output, space=args.space,
                                  start_frame=args.start_frame, max_frames=args.max_frames, preview=args.preview,
                                  max_speed=args.max_speed, gate_slack=args.gate_slack,
                                  reset_after=args.reset_after, calibration_size=args.calibration_size,
                                  provenance=provenance, pitch_region=region, off_field_grace=args.off_field_grace,
                                  static_seconds=args.outside_static_seconds,
                                  save_candidates=args.save_candidates and args.candidate_cache is None)
        finally:
            if args.candidate_cache:
                detector.close()
        print(f"完成：{summary['frames']} 帧，{summary['detected']} 帧有检测结果；输出目录 {args.output}")
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as error:
        p.exit(2, f"错误：{error}\n")
    except KeyboardInterrupt:
        p.exit(130, "处理已停止；重跑同一 batch 命令可跳过已完成的半场。\n")
