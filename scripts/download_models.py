#!/usr/bin/env python3
"""Download selected official YOLO detection weights; Python standard library only.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models yolo26l yolo26m
    python scripts/download_models.py --dry-run

Asset sizes and SHA-256 digests were checked against the official GitHub release
metadata on 2026-11-02. No datasets, packages or inference assets are downloaded.
"""

import argparse
from dataclasses import dataclass
import hashlib
from http.client import HTTPException
import math
from pathlib import Path
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


RELEASE = "v8.4.0"
BASE_URL = f"https://github.com/ultralytics/assets/releases/download/{RELEASE}"


@dataclass(frozen=True)
class ModelAsset:
    size: int
    sha256: str


MODELS = {
    "yolo26x": ModelAsset(118667365, "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"),
    "yolo26l": ModelAsset(53211173, "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26"),
    "yolo26m": ModelAsset(44255705, "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"),
    "yolo11x": ModelAsset(114636239, "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"),
}


def verify(path: Path, asset: ModelAsset) -> None:
    if path.stat().st_size != asset.size:
        raise ValueError(f"文件大小不匹配：{path.name}，预期 {asset.size} 字节")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != asset.sha256:
        raise ValueError(f"SHA-256 校验失败：{path.name}")


def download(model: str, output: Path, *, timeout=60., retries=3, force=False) -> Path:
    if retries < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("retries 必须大于等于 1，timeout 必须为正有限数")
    asset = MODELS[model]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{model}.pt"
    if target.exists():
        try:
            verify(target, asset)
            print(f"已存在且校验通过，跳过：{target}")
            return target
        except ValueError as error:
            if not force:
                raise ValueError(f"{error}；原文件已保留，如需重新下载请加 --force") from error
    url = f"{BASE_URL}/{model}.pt"
    for attempt in range(1, retries + 1):
        temporary = None
        try:
            request = Request(url, headers={"User-Agent": "TactiVision-model-downloader/0.1"})
            print(f"下载 {model}（{asset.size / 1024**2:.1f} MiB），尝试 {attempt}/{retries}", flush=True)
            with tempfile.NamedTemporaryFile(dir=output, prefix=f".{model}.", suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                with urlopen(request, timeout=timeout) as response:
                    received = 0
                    next_report = 10 * 1024 * 1024
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > asset.size:
                            raise ValueError("响应内容超出官方权重大小")
                        stream.write(chunk)
                        if received >= next_report:
                            print(f"  {100 * received / asset.size:.0f}%", flush=True)
                            next_report += 10 * 1024 * 1024
            verify(temporary, asset)
            temporary.replace(target)
            print(f"完成，SHA-256 校验通过：{target}")
            return target
        except (OSError, ValueError, HTTPException) as error:
            if attempt == retries:
                raise RuntimeError(f"{model} 下载失败：{error}") from error
            print(f"  下载或校验失败：{error}；即将重试", file=sys.stderr)
            time.sleep(min(2 ** (attempt - 1), 5))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError("下载未完成")


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载官方 YOLO 检测权重，默认精度优先 YOLO26x")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=["yolo26x"])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    parser.add_argument("--timeout", type=float, default=60., help="单次网络读取超时秒数")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="成功下载校验后替换已有不匹配文件")
    parser.add_argument("--dry-run", action="store_true", help="仅展示下载计划，不联网、不创建文件")
    args = parser.parse_args(argv)
    if args.retries < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--retries 必须大于等于 1，--timeout 必须为正有限数")
    try:
        for model in dict.fromkeys(args.models):
            asset = MODELS[model]
            if args.dry_run:
                print(f"{BASE_URL}/{model}.pt\n  -> {args.output / (model + '.pt')}"
                      f"\n  {asset.size} bytes | sha256:{asset.sha256}")
            else:
                download(model, args.output, timeout=args.timeout, retries=args.retries, force=args.force)
    except (OSError, RuntimeError, ValueError, URLError) as error:
        parser.exit(1, f"错误：{error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\n下载已中断，未完成的临时文件已清理。\n")


if __name__ == "__main__":
    main()
