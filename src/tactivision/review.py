"""Audit saved detections and render review artifacts without running YOLO again."""

from collections import Counter
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import subprocess
import shutil

import cv2
import numpy as np

from .pipeline import file_hash
from .render import Preview


VALUE_KEYS = ("u_px", "v_px", "x_m", "y_m", "confidence", "track_id",
              "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
STATUS = {"detected": 1, "missing": 2, "rejected": 3}


def longest_run(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    if not len(changes):
        return (0, 0)
    lengths = changes[1::2] - changes[::2]
    k = int(np.argmax(lengths))
    return int(changes[2 * k]), int(changes[2 * k + 1])


def stationary_run(values, valid, tolerance=2.):
    """Longest contiguous same-track run within tolerance pixels of its first centre."""
    best, start, anchor = (0, 0), None, None
    for i in range(len(valid)):
        if not valid[i]:
            if start is not None and i - start > best[1] - best[0]:
                best = (start, i)
            start, anchor = None, None
            continue
        u, v, track = values[i, 0], values[i, 1], values[i, 5]
        if anchor is None or track != anchor[2] or (u - anchor[0]) ** 2 + (v - anchor[1]) ** 2 > tolerance ** 2:
            if start is not None and i - start > best[1] - best[0]:
                best = (start, i)
            start, anchor = i, (u, v, track)
    if start is not None and len(valid) - start > best[1] - best[0]:
        best = (start, len(valid))
    return best


def analyze_half(directory, expected_frames=None, samples=4):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    marker = json.loads((directory / "completed.json").read_text())
    n, fps = int(summary["frames"]), float(summary["fps"])
    if n <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid frame count/FPS in {directory}")
    errors = []
    def error(message):
        if len(errors) < 20:
            errors.append(message)
    for name, digest in marker.get("sha256", {}).items():
        path = directory / name
        if path.parent != directory or not path.is_file() or file_hash(path) != digest:
            error(f"Artifact checksum mismatch: {name}")
    for required in ("ball_positions.csv", "summary.json", "calibration_report.json"):
        if required not in marker.get("sha256", {}):
            error(f"Completion marker missing checksum: {required}")
    if expected_frames is not None and n != expected_frames:
        error(f"Expected {expected_frames} source frames, summary has {n}")
    if marker.get("frames") != n:
        error("Completion marker frame count disagrees with summary")
    values = np.full((n, len(VALUE_KEYS)), np.nan)
    states = np.zeros(n, dtype=np.uint8)
    candidates = np.zeros(n, dtype=int)
    length, width = summary["coordinates"]["pitch_size"]
    image_width, image_height = summary["image_size"]
    rows = 0
    with (directory / "ball_positions.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {*VALUE_KEYS, "frame", "timestamp_ms", "status", "candidate_count", "in_pitch"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required CSV columns in {directory}")
        for i, row in enumerate(reader):
            rows += 1
            if i >= n:
                error("CSV has more rows than summary")
                continue
            try:
                frame = int(row["frame"])
                if frame != int(summary["start_frame"]) + i:
                    error(f"Non-contiguous frame index at CSV row {i}")
                timestamp = float(row["timestamp_ms"])
                if not math.isfinite(timestamp) or abs(timestamp - frame / fps * 1000) > .001:
                    error(f"Timestamp mismatch at frame {frame}")
                states[i] = STATUS.get(row["status"], 0)
                if states[i] == 0:
                    error(f"Unknown status at frame {frame}")
                candidates[i] = int(row["candidate_count"])
                if states[i] == 1:
                    values[i] = [float(row[key]) for key in VALUE_KEYS]
                    u, v, x, y, confidence, track, x1, y1, x2, y2 = values[i]
                    if not np.isfinite(values[i]).all() or not 0 <= confidence <= 1:
                        error(f"Invalid detected values at frame {frame}")
                    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
                        error(f"Invalid image bbox at frame {frame}")
                    if abs(u - (x1 + x2) / 2) > .001 or abs(v - (y1 + y2) / 2) > .001:
                        error(f"BBox centre mismatch at frame {frame}")
                    if candidates[i] < 1 or track < 1 or track != int(track):
                        error(f"Invalid candidate count/track id at frame {frame}")
                    inside = 0 <= x <= length and 0 <= y <= width
                    if row["in_pitch"] != str(inside):
                        error(f"in_pitch mismatch at frame {frame}")
                elif any(row[key] for key in VALUE_KEYS):
                    error(f"Missing/rejected frame contains detected values: {frame}")
            except (ValueError, TypeError, OverflowError):
                error(f"Unparseable CSV row {i}")
    if rows != n:
        error(f"CSV row count {rows} differs from summary count {n}")
    counts = {name: int(np.sum(states == code)) for name, code in STATUS.items()}
    for name, count in counts.items():
        if summary.get(name) != count:
            error(f"{name} count disagrees with summary")
    valid = (states == 1) & np.isfinite(values).all(axis=1)
    indices = np.flatnonzero(valid)
    inside = valid & (values[:, 2] >= 0) & (values[:, 2] <= length) & (values[:, 3] >= 0) & (values[:, 3] <= width)
    stationary = stationary_run(values, valid)
    gap = longest_run(~valid)
    adjacent = valid[1:] & valid[:-1] & (values[1:, 5] == values[:-1, 5])
    pixel_steps = np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)
    ground_steps = np.linalg.norm(np.diff(values[:, 2:4], axis=0), axis=1)
    dominant = 0.
    if len(indices):
        _, cell_counts = np.unique(np.floor(values[valid, :2] / 4).astype(int), axis=0, return_counts=True)
        dominant = float(cell_counts.max() / len(indices))
    outside_fraction = float(1 - inside.sum() / len(indices)) if len(indices) else None
    flags = []
    if (stationary[1] - stationary[0]) / fps >= 30:
        flags.append("stationary_target_30s")
    if outside_fraction is not None and outside_fraction >= .5:
        flags.append("mostly_outside_pitch")
    if dominant >= .5:
        flags.append("dominant_4px_cell")
    if (gap[1] - gap[0]) / fps >= 2:
        flags.append("gap_2s")
    sample_indices = set(int(i) for i in np.linspace(0, n - 1, samples))
    if stationary[1] > stationary[0]:
        sample_indices.add((stationary[0] + stationary[1] - 1) // 2)
    if adjacent.any():
        sample_indices.add(int(np.nanargmax(np.where(adjacent, pixel_steps, -1))) + 1)
    sample_indices = sorted(sample_indices)
    key = f"{directory.parent.name}/{directory.name}"
    metadata = {
        "key": key, "match": directory.parent.name, "half": directory.name,
        "frames": n, "csv_rows": rows, "fps": fps, "duration_s": n / fps,
        "integrity_ok": not errors, "integrity_errors": errors, **counts,
        "coverage": counts["detected"] / n,
        "inside_detected_fraction": float(inside.sum() / len(indices)) if len(indices) else None,
        "outside_detected_fraction": outside_fraction,
        "confidence_median": float(np.median(values[valid, 4])) if len(indices) else None,
        "track_segments": int(len(np.unique(values[valid, 5]))),
        "dominant_4px_cell_fraction": dominant,
        "longest_stationary_s": (stationary[1] - stationary[0]) / fps,
        "stationary_frame_range": [stationary[0], stationary[1]],
        "longest_gap_s": (gap[1] - gap[0]) / fps,
        "max_apparent_step_m": float(ground_steps[adjacent].max()) if adjacent.any() else None,
        "flags": flags, "samples": sample_indices,
        "video": summary["video"], "source_output": str(directory.resolve()),
        "pitch_size": [length, width],
    }
    # Keep a bounded visual sample; gaps/track changes are retained as segment ids.
    plot_indices = indices[np.unique(np.linspace(0, len(indices) - 1, min(1000, len(indices))).astype(int))] if len(indices) else []
    metadata["trajectory"] = [[int(i), *np.round(values[i, [2, 3]], 3).tolist(), int(values[i, 5])]
                              for i in plot_indices]
    metadata["minute_coverage"] = []
    for first in range(0, n, max(1, round(fps * 60))):
        stop = min(n, first + round(fps * 60))
        metadata["minute_coverage"].append(round(float(valid[first:stop].mean() * 100), 2))
    return metadata, values, states, summary


def row_at(index, values, states, summary):
    row = dict.fromkeys(VALUE_KEYS, "")
    row.update(frame=int(summary["start_frame"]) + index,
               timestamp_ms=(int(summary["start_frame"]) + index) / summary["fps"] * 1000,
               status={v: k for k, v in STATUS.items()}.get(int(states[index]), "missing"))
    if states[index] == 1 and np.isfinite(values[index]).all():
        row.update(zip(VALUE_KEYS, values[index]))
        length, width = summary["coordinates"]["pitch_size"]
        row["in_pitch"] = bool(0 <= row["x_m"] <= length and 0 <= row["y_m"] <= width)
    else:
        row["status"] = "missing"
    return row


def trajectory_svg(metadata, path):
    """Export a small, labelled plot; retain all numerical data in the source CSV."""
    L, W = metadata["pitch_size"]
    sx, sy, ox, oy = 6., 6., 80., 60.
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="860" height="650" viewBox="0 0 860 650">',
             '<rect width="860" height="650" fill="#151b22"/>',
             f'<text x="40" y="30" fill="white" font-family="sans-serif" font-size="18">{metadata["key"]} — selected ground-plane positions</text>',
             f'<rect x="{ox}" y="{oy}" width="{L*sx}" height="{W*sy}" fill="#254437" stroke="#bdcdc5"/>',
             f'<path d="M {ox+L*sx/2} {oy} v {W*sy}" stroke="#bdcdc5"/>',
             f'<ellipse cx="{ox+L*sx/2}" cy="{oy+W*sy/2}" rx="{9.15*sx}" ry="{9.15*sy}" fill="none" stroke="#bdcdc5"/>']
    hidden = 0
    for frame, x, y, track in metadata["trajectory"]:
        if -10 <= x <= L + 10 and -5 <= y <= W + 12:
            lines.append(f'<circle cx="{ox+x*sx:.2f}" cy="{oy+y*sy:.2f}" r="2" fill="#f4bd50" opacity=".55"><title>frame {frame}, track {track}, x={x}, y={y} m</title></circle>')
        else:
            hidden += 1
    for x in (0, L/2, L):
        lines.append(f'<text x="{ox+x*sx}" y="{oy+W*sy+95}" fill="#ddd" font-family="sans-serif">{x:g}</text>')
    for y in (0, W/2, W):
        lines.append(f'<text x="30" y="{oy+y*sy}" fill="#ddd" font-family="sans-serif">{y:g}</text>')
    for y, text in [(oy+W*sy+125, "Pitch x (m), corner origin; y increases toward near sideline"),
                    (oy+W*sy+150, f"Source: ball_positions.csv, full half; up to 1000 sampled points; {hidden} outside plot window"),
                    (oy+W*sy+175, "Yellow: selected positions; coverage is not accuracy; airborne height is not recovered")]:
        lines.append(f'<text x="40" y="{y}" fill="#ddd" font-size="12" font-family="sans-serif">{text}</text>')
    lines += ['<text x="10" y="45" fill="#ddd" font-family="sans-serif">y (m)</text>', '</svg>']
    Path(path).write_text("\n".join(lines))


def render_review(metadata, values, states, summary, destination, clip_seconds=4.):
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(summary["video"])
    if not capture.isOpened():
        raise ValueError(f"Cannot read source video: {summary['video']}")
    try:
        tiles = []
        for i in metadata["samples"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(summary["start_frame"]) + i)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode review frame {i}")
            row = row_at(i, values, states, summary)
            tile = np.full((360, 640, 3), (25, 23, 21), np.uint8)
            cv2.putText(tile, f"{metadata['key']} frame={row['frame']} t={row['timestamp_ms']/1000:.1f}s",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 230, 230), 1)
            original = image.copy()
            if row["status"] == "detected":
                box = [round(row[key]) for key in VALUE_KEYS[6:]]
                cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 220, 255), 3)
            scale = min(640 / image.shape[1], 180 / image.shape[0])
            small = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
            tile[35:35+small.shape[0], :small.shape[1]] = small
            cv2.putText(tile, f"Selected status: {row['status']}", (12, 247), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 230, 230), 1)
            if row["status"] == "detected":
                u, v = round(row["u_px"]), round(row["v_px"])
                crop = original[max(0,v-45):min(image.shape[0],v+45), max(0,u-70):min(image.shape[1],u+70)]
                if crop.size:
                    tile[224:350, 432:628] = cv2.resize(crop, (196,126))
                cv2.putText(tile, f"Confidence={row['confidence']:.3f}", (12,277), cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1)
                cv2.putText(tile, f"Pitch=({row['x_m']:.1f}, {row['y_m']:.1f}) m", (12,307),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1)
                cv2.putText(tile, "IN pitch" if row["in_pitch"] else "OUTSIDE pitch", (12,337),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),1)
            tiles.append(tile)
        columns = 2
        sheet = np.full((math.ceil(len(tiles)/columns)*360, columns*640, 3), (25,23,21), np.uint8)
        for k,tile in enumerate(tiles):
            sheet[(k//columns)*360:(k//columns+1)*360,(k%columns)*640:(k%columns+1)*640] = tile
        if not cv2.imwrite(str(destination / "contact_sheet.jpg"), sheet):
            raise ValueError("Cannot write contact sheet")
        if clip_seconds > 0:
            stationary = metadata["stationary_frame_range"]
            centre = (stationary[0]+stationary[1])//2 if stationary[1]>stationary[0] else metadata["frames"]//2
            count = min(metadata["frames"], max(1,round(clip_seconds*metadata["fps"])))
            first = min(max(0,centre-count//2),metadata["frames"]-count)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(summary["start_frame"])+first)
            calibration = type("Pitch", (), dict(pitch_length=metadata["pitch_size"][0], pitch_width=metadata["pitch_size"][1]))()
            renderer = Preview(calibration,metadata["fps"])
            writer = None
            temporary = destination / "review_raw.mp4"
            try:
                for i in range(first,first+count):
                    ok,frame = capture.read()
                    if not ok:
                        raise ValueError("Incomplete review clip decoding")
                    rendered=renderer.draw(frame,row_at(i,values,states,summary))
                    if i == first:
                        cv2.imwrite(str(destination / "review_poster.jpg"), rendered)
                    if writer is None:
                        writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*"mp4v"),metadata["fps"],(rendered.shape[1],rendered.shape[0]))
                        if not writer.isOpened(): raise ValueError("Cannot encode review clip")
                    writer.write(rendered)
            finally:
                if writer is not None: writer.release()
            final=destination/"review.mp4"
            if shutil.which("ffmpeg"):
                subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(temporary),
                                "-c:v","libx264","-preset","veryfast","-crf","23","-pix_fmt","yuv420p",
                                "-movflags","+faststart",str(final)],check=True,capture_output=True)
                temporary.unlink()
            else:
                temporary.replace(final)
            metadata["review_clip"]={"first_frame":first,"frames":count,"seconds":count/metadata["fps"]}
    finally:
        capture.release()


def run_review(source, output, samples=4, clip_seconds=4.):
    source, output = Path(source).resolve(), Path(output).resolve()
    if samples < 2 or not math.isfinite(clip_seconds) or clip_seconds < 0:
        raise ValueError("samples must be at least 2 and clip-seconds nonnegative")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Review output must be separate from the reconstruction directory")
    manifest = json.loads((source / "batch_status.json").read_text())
    if not manifest.get("jobs"):
        raise ValueError("The selected batch has no video jobs")
    if any(job["status"] not in ("completed","skipped") for job in manifest["jobs"].values()):
        raise ValueError("The selected batch contains unfinished or failed jobs")
    output.mkdir(parents=True,exist_ok=True)
    halves=[]
    for name, job in sorted(manifest["jobs"].items()):
        parts=Path(name).parts
        if len(parts)!=2 or not parts[0].isdigit() or parts[1] not in ("1st","2nd"):
            raise ValueError(f"Invalid match/half key in batch manifest: {name}")
        directory=source/name
        metadata,values,states,summary=analyze_half(directory,expected_frames=job["frames"],samples=samples)
        destination=output/name
        destination.mkdir(parents=True,exist_ok=True)
        trajectory_svg(metadata,destination/"trajectory.svg")
        render_review(metadata,values,states,summary,destination,clip_seconds=clip_seconds)
        (destination/"metrics.json").write_text(json.dumps(metadata,indent=2,allow_nan=False))
        halves.append(metadata)
        print(f"Reviewed {name}: integrity={metadata['integrity_ok']}, coverage={metadata['coverage']:.1%}, "
              f"outside={metadata['outside_detected_fraction']}, stationary={metadata['longest_stationary_s']:.1f}s",flush=True)
    report={"source":str(source),"generated_at":datetime.now(timezone.utc).isoformat(),
            "half_count":len(halves),"match_count":len({h['match'] for h in halves}),
            "frames":sum(h['frames'] for h in halves),"detected":sum(h['detected'] for h in halves),
            "integrity_passed":sum(h['integrity_ok'] for h in halves),
            "flagged_halves":sum(bool(h['flags']) for h in halves),
            "notes":["Coverage is not accuracy. No independent ball ground truth was used.",
                     "Stationarity and outside-pitch flags identify review candidates, not automatic false positives.",
                     "Source CSV, official calibration and completion markers were not modified.",
                     "Trajectory plots show sampled selected ground-plane positions, without interpolation or smoothing."],
            "halves":halves}
    (output/"quality_report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    fields=[k for k in halves[0] if k not in ("trajectory","minute_coverage","samples","review_clip","integrity_errors","stationary_frame_range")]
    with (output/"quality_summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        for half in halves: writer.writerow(half)
    write_html_report(report,output/"index.html")
    print(f"Review complete: {len(halves)} halves, {report['frames']:,} frames -> {output}",flush=True)
    return report


def write_html_report(report, destination):
    """An offline viewer with relative image/video links and embedded metrics."""
    data=json.dumps(report,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TactiVision · 重建结果质量检查</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#111820;color:#e3e9ef}
body{margin:0 auto;padding:28px;max-width:1400px}h1{font-size:24px;margin:0 0 10px}h2{font-size:18px}
p{line-height:1.7;color:#b8c4cf}section{margin:24px 0}.notice{border-left:3px solid #d6a647;padding:10px 18px;background:#1c252d}
.stats{display:flex;gap:36px;flex-wrap:wrap;margin:24px 0}.stats b{display:block;font-size:24px}.stats span{font-size:13px;color:#9aaebd}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}select,button{background:#23313c;color:#e3e9ef;border:1px solid #44515e;border-radius:4px;padding:8px}
img,video{display:block;width:100%;margin-top:14px}svg{width:100%;background:#17242b}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:10px 8px;text-align:left;border-bottom:1px solid #31404b}th{color:#99b3c4}caption{text-align:left;color:#99b3c4;margin:10px 0}
a{color:#83b9e2}.muted{font-size:12px;color:#99adbd}.scroll{overflow:auto}@media(max-width:900px){.columns{grid-template-columns:1fr}}
</style>
<h1>TactiVision · 轨迹与检测质量复查</h1>
<p id="source"></p>
<div class="notice">检测覆盖率记录“选中了候选”的帧数比例。长期静止或投影在界外的轨迹需要回看画面，确认是否选中了场边备用球。高覆盖率不能作为比赛用球跟踪准确率；自动标记用于安排人工复查，不作为误检真值。</div>
<div class="stats" id="stats"></div>
<section><label for="half">选择比赛 / 半场：</label><select id="half"></select><p id="detail"></p>
<div class="columns"><div><h2>球场平面上的已选位置</h2><svg id="pitch" viewBox="0 0 800 610" aria-label="球场米制坐标图"></svg>
<p class="muted">x、y 单位：米；角点原点，y 朝近侧边线增大。每半场最多等距抽取 1000 个有效点，点位不插值、不平滑；图外点数另列。强调色为球场内位置，灰色为球场外位置。</p>
<h2>每分钟检测覆盖率</h2><svg id="coverage" viewBox="0 0 800 220" aria-label="每分钟检测覆盖率"></svg>
<p class="muted">横轴：该半场视频起点后的分钟；纵轴：该分钟有选中候选的帧数比例（%）。数据来自整个半场。</p>
</div><div><h2>抽检短片</h2><video id="video" controls preload="metadata"></video><p class="muted" id="clip"></p>
<a id="svg-link">打开独立轨迹 SVG</a> · <a id="metrics-link">打开该半场指标 JSON</a></div></div>
<h2>原视频抽帧与选中目标放大</h2><img id="contact" alt="多个时刻的原视频、所选检测框及目标局部放大图">
<p class="muted">抽检样本包含均匀时刻、最长像素静止段中点和最大同轨相邻帧像素跳变处。短片围绕最长像素静止段中点截取；没有独立人工足球真值，不报告精确率、召回率或米制定位准确率。</p></section>
<section><h2>全部半场汇总</h2><div class="scroll"><table><caption>覆盖率与投影位置统计来自保存的 CSV；静止指同轨连续像素中心距该段起点不超过 2 px。</caption>
<thead><tr><th>半场</th><th>完整性</th><th>总帧数</th><th>检测覆盖率</th><th>已选位置投影界外</th><th>最长像素静止</th><th>最长缺失</th><th>置信度中位数</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<script>
const report=__DATA__;
const $=id=>document.getElementById(id), pct=x=>x===null?'—':(100*x).toFixed(1)+'%';
const add=(parent,tag,text)=>{const node=document.createElement(tag);node.textContent=text;parent.append(node);return node};
const shape=(parent,tag,attrs)=>{const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))node.setAttribute(k,v);parent.append(node);return node};
$('source').textContent='源目录：'+report.source+' · '+report.match_count+' 场 / '+report.half_count+' 半场 · 报告生成时间（UTC）：'+report.generated_at;
const inside=report.halves.reduce((sum,h)=>sum+h.detected*(h.inside_detected_fraction||0),0);
for(const[value,label]of [[report.integrity_passed+'/'+report.half_count,'文件完整性通过'],[report.frames.toLocaleString(),'核验帧数'],[pct(report.detected/report.frames),'检测覆盖率'],[pct(inside/report.detected),'已选位置投影在球场内']]){const item=add($('stats'),'div','');add(item,'b',value);add(item,'span',label)}
for(const half of report.halves){const option=add($('half'),'option',half.key);option.value=half.key;
const row=add($('rows'),'tr','');const button=add(add(row,'td',''),'button',half.key);button.onclick=()=>{$('half').value=half.key;draw();window.scrollTo({top:0,behavior:'smooth'})};
for(const text of [half.integrity_ok?'通过':'需检查',half.frames.toLocaleString(),pct(half.coverage),pct(half.outside_detected_fraction),half.longest_stationary_s.toFixed(1)+' s',half.longest_gap_s.toFixed(1)+' s',half.confidence_median===null?'—':half.confidence_median.toFixed(3)])add(row,'td',text)}
function draw(){const h=report.halves.find(h=>h.key===$('half').value),L=h.pitch_size[0],W=h.pitch_size[1];
$('detail').textContent=h.key+' · '+(h.duration_s/60).toFixed(1)+' 分钟 · '+h.track_segments+' 个局部关联段 · 投影界外 '+pct(h.outside_detected_fraction)+' · 最长像素静止 '+h.longest_stationary_s.toFixed(1)+' 秒';
$('contact').src=h.key+'/contact_sheet.jpg';$('video').src=h.key+'/review.mp4';$('video').poster=h.key+'/review_poster.jpg';$('svg-link').href=h.key+'/trajectory.svg';$('metrics-link').href=h.key+'/metrics.json';
$('video').hidden=!h.review_clip;$('clip').textContent=h.review_clip?'起始帧 '+h.review_clip.first_frame+'，共 '+h.review_clip.frames+' 帧 / '+h.review_clip.seconds.toFixed(1)+' 秒。':'本次仅生成抽帧图像。';
const p=$('pitch');p.replaceChildren();shape(p,'rect',{x:70,y:50,width:L*6,height:W*6,fill:'#254437',stroke:'#b5c9be'});shape(p,'line',{x1:70+L*3,y1:50,x2:70+L*3,y2:50+W*6,stroke:'#b5c9be'});shape(p,'circle',{cx:70+L*3,cy:50+W*3,r:9.15*6,fill:'none',stroke:'#b5c9be'});
let hidden=0;for(const[frame,x,y,track]of h.trajectory){if(x<-10||x>L+10||y<-5||y>W+12){hidden++;continue}const c=shape(p,'circle',{cx:70+x*6,cy:50+y*6,r:2.5,fill:x>=0&&x<=L&&y>=0&&y<=W?'#7ebcdf':'#b1b7ba',opacity:.6});shape(c,'title',{}).textContent='frame '+frame+' · track '+track+' · ('+x+', '+y+') m'}
for(const x of [0,L/2,L])shape(p,'text',{x:70+x*6,y:550,fill:'#b8c4cf','font-size':14}).textContent=x;for(const y of [0,W/2,W])shape(p,'text',{x:22,y:50+y*6,fill:'#b8c4cf','font-size':14}).textContent=y;
shape(p,'text',{x:330,y:580,fill:'#b8c4cf'}).textContent='球场 x (m)';shape(p,'text',{x:8,y:28,fill:'#b8c4cf'}).textContent='y (m)';shape(p,'text',{x:420,y:28,fill:'#b8c4cf','font-size':12}).textContent=hidden+' 个抽样点在显示范围外';
const c=$('coverage');c.replaceChildren();for(const value of [0,50,100]){shape(c,'line',{x1:50,y1:170-value*1.35,x2:770,y2:170-value*1.35,stroke:'#31404b'});shape(c,'text',{x:8,y:174-value*1.35,fill:'#b8c4cf','font-size':12}).textContent=value+'%'}
const step=720/h.minute_coverage.length;h.minute_coverage.forEach((v,i)=>shape(c,'rect',{x:50+i*step,y:170-v*1.35,width:Math.max(1,step-2),height:v*1.35,fill:'#7ebcdf'}));for(const i of [0,Math.floor(h.minute_coverage.length/2),h.minute_coverage.length-1])shape(c,'text',{x:50+i*step,y:192,fill:'#b8c4cf','font-size':12}).textContent=i;shape(c,'text',{x:330,y:215,fill:'#b8c4cf','font-size':12}).textContent='视频时间（分钟）';}
$('half').value=report.halves.some(h=>h.key==='117093/1st')?'117093/1st':report.halves[0].key;$('half').onchange=draw;draw();
</script></html>'''
    Path(destination).write_text(html.replace('__DATA__',data))
