#!/usr/bin/env python3
"""Run the corrected dual-GPU batch in a daily daytime window (Linux)."""

import argparse
from datetime import datetime, time, timedelta
import fcntl
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clock_time(value):
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise argparse.ArgumentTypeError("时间格式必须为 HH:MM，例如 08:30") from error


def parser():
    p = argparse.ArgumentParser(description="每天定时启动/停止双卡比赛用球重建；完成的半场自动跳过")
    p.add_argument("--start", type=clock_time, default=time(8, 30))
    p.add_argument("--stop", type=clock_time, default=time(16, 30))
    p.add_argument("--timezone", default="Asia/Shanghai")
    p.add_argument("--stop-timeout", type=float, default=60., help="优雅停止超时秒数，之后强制清理进程组")
    p.add_argument("--weights", type=Path, default=Path("models/yolo26x.pt"))
    p.add_argument("--dataset", type=Path, default=Path("datasets/SoccerTrack-v2/data"))
    p.add_argument("--output", type=Path, default=Path("outputs/all_matches_v2"))
    p.add_argument("--devices", nargs="+", default=["0", "1"])
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="仅显示时间窗口和批处理命令，不运行、不创建输出")
    return p


def project_path(path):
    return (PROJECT_ROOT / path).resolve()


def batch_command(args):
    command = [sys.executable, "-u", "-m", "tactivision", "batch",
               "--dataset", str(project_path(args.dataset)),
               "--weights", str(project_path(args.weights)),
               "--output", str(project_path(args.output)),
               "--devices", *args.devices, "--cpu-threads", str(args.cpu_threads),
               "--tile-size", str(args.tile_size), "--imgsz", str(args.imgsz),
               "--overlap", "0.2", "--pitch-region", "keypoints", "--focus-pitch",
               "--off-field-grace", "1.0", "--outside-static-seconds", "1.0", "--save-candidates"]
    if args.preview:
        command.append("--preview")
    return command


def window_at(now, start, stop):
    """Return whether now is in [start, stop), and the next scheduled start."""
    begins = datetime.combine(now.date(), start, tzinfo=now.tzinfo)
    ends = datetime.combine(now.date(), stop, tzinfo=now.tzinfo)
    active = begins <= now < ends
    return active, begins if now < begins else begins + timedelta(days=1)


def log(message):
    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)


def kill_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def stop_batch(process, timeout):
    # Signal the parent first so batch can publish interrupted status and join workers.
    if process.poll() is None:
        log(f"停止 batch PID={process.pid}，等待双卡子进程退出")
        try:
            process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log("优雅停止超时，强制停止本次 batch 的整个进程组")
    # Also remove orphaned GPU workers if the parent failed or exited unexpectedly.
    kill_group(process.pid, signal.SIGKILL)
    process.wait()


def run_schedule(args, zone, command, shutdown, *, now=None, launch=None):
    now = now or (lambda: datetime.now(zone))
    launch = launch or subprocess.Popen
    attempted_date = None
    process = None
    output = project_path(args.output)
    try:
        with (output / "batch.log").open("a", buffering=1) as batch_log:
            while not shutdown.is_set():
                current = now()
                active, _ = window_at(current, args.start, args.stop)
                if process is not None and process.poll() is not None:
                    log(f"batch 已退出，退出码={process.returncode}；等待下个运行日")
                    stop_batch(process, args.stop_timeout)
                    process = None
                if process is not None and not active:
                    stop_batch(process, args.stop_timeout)
                    process = None
                    log("今日运行窗口结束，等待次日启动")
                if active and process is None and attempted_date != current.date():
                    attempted_date = current.date()
                    log(f"启动双卡 batch：{shlex.join(command)}")
                    try:
                        process = launch(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                                         stdout=batch_log, stderr=subprocess.STDOUT, start_new_session=True)
                    except OSError as error:
                        log(f"启动失败：{error}；等待下个运行日，请检查配置")
                shutdown.wait(1.)
    finally:
        if process is not None:
            stop_batch(process, args.stop_timeout)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.start >= args.stop:
        p.error("要求 --start 早于 --stop（仅支持同一天内的运行窗口）")
    if not math.isfinite(args.stop_timeout) or args.stop_timeout <= 0:
        p.error("--stop-timeout 必须为正有限数")
    if args.cpu_threads <= 0 or args.tile_size < 0 or args.imgsz <= 0:
        p.error("--cpu-threads 和 --imgsz 必须为正，--tile-size 不得小于 0")
    if not args.devices or len(set(args.devices)) != len(args.devices) or any(not d.isdigit() for d in args.devices):
        p.error("--devices 必须是不同的 GPU 编号，例如 0 1")
    try:
        zone = ZoneInfo(args.timezone)
    except ZoneInfoNotFoundError:
        p.error(f"未知时区：{args.timezone}")
    command = batch_command(args)
    current = datetime.now(zone)
    active, next_start = window_at(current, args.start, args.stop)
    print(f"每天 {args.start:%H:%M}–{args.stop:%H:%M}，时区 {zone.key}；"
          f"{'启动后立即运行' if active else '等待至 ' + next_start.isoformat(timespec='minutes')}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not project_path(args.weights).is_file():
        p.error(f"找不到权重：{project_path(args.weights)}")
    if not project_path(args.dataset).is_dir():
        p.error(f"找不到数据目录：{project_path(args.dataset)}")
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shutdown = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: shutdown.set())
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (output / "schedule.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                p.error(f"该输出目录已有定时脚本运行：{output}")
            pid_file = output / "schedule.pid"
            pid_file.write_text(str(os.getpid()) + "\n")
            try:
                log("定时脚本就绪；Ctrl+C 或 SIGTERM 会同时停止当前 batch")
                run_schedule(args, zone, command, shutdown)
            finally:
                pid_file.unlink(missing_ok=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
