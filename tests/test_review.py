import csv
import json
from pathlib import Path

import numpy as np
import pytest
import cv2

from tactivision.pipeline import FIELDS, file_hash
from tactivision.review import analyze_half, longest_run, stationary_run, trajectory_svg, write_html_report, render_review


def saved_half(tmp_path, count=100, missing=()):
    directory = tmp_path / "117093" / "1st"
    directory.mkdir(parents=True)
    detected = count - len(missing)
    summary = dict(frames=count, fps=25., start_frame=0, image_size=[640,480],
                   detected=detected, missing=len(missing), rejected=0,
                   coordinates={"pitch_size":[105.,68.]}, video="unused.mp4")
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "calibration_report.json").write_text("{}")
    with (directory / "ball_positions.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()
        for frame in range(count):
            row=dict.fromkeys(FIELDS,"")
            row.update(frame=frame,timestamp_ms=frame*40,status="missing" if frame in missing else "detected",candidate_count=0 if frame in missing else 1)
            if frame not in missing:
                row.update(track_id=1,confidence=.7,u_px=100.,v_px=110.,undistorted_u_px=100.,undistorted_v_px=110.,
                           x_m=48.,y_m=70.,x_center_m=-4.5,y_center_m=-36.,in_pitch=False,
                           bbox_x1=96.,bbox_y1=106.,bbox_x2=104.,bbox_y2=114.)
            writer.writerow(row)
    names=("summary.json","ball_positions.csv","calibration_report.json")
    (directory/"completed.json").write_text(json.dumps({"frames":count,"sha256":{n:file_hash(directory/n) for n in names}}))
    return directory


def test_saved_integrity_and_static_outside_target_are_distinct(tmp_path):
    directory=saved_half(tmp_path,count=1000)
    report,values,states,summary=analyze_half(directory,1000)
    assert report["integrity_ok"]
    assert report["coverage"]==1
    assert report["inside_detected_fraction"]==0
    assert report["longest_stationary_s"]==40
    assert set(report["flags"])=={"stationary_target_30s","mostly_outside_pitch","dominant_4px_cell"}
    assert report["confidence_median"]==.7
    assert report["trajectory"][0][1:3]==[48.,70.]
    assert file_hash(directory/"ball_positions.csv")==json.loads((directory/"completed.json").read_text())["sha256"]["ball_positions.csv"]


def test_gap_boundaries_and_stationary_segments_do_not_bridge_missing_frames(tmp_path):
    directory=saved_half(tmp_path,count=100,missing=tuple(range(20,70)))
    report,*_=analyze_half(directory,100)
    assert report["detected"]==50 and report["missing"]==50
    assert report["coverage"]==.5
    assert report["longest_gap_s"]==2
    assert report["stationary_frame_range"]==[70,100]
    assert "gap_2s" in report["flags"]


def test_modified_csv_and_frame_order_fail_integrity(tmp_path):
    directory=saved_half(tmp_path)
    path=directory/"ball_positions.csv"
    lines=path.read_text().splitlines(); lines[1],lines[2]=lines[2],lines[1]
    path.write_text("\n".join(lines)+"\n")
    report,*_=analyze_half(directory,100)
    assert not report["integrity_ok"]
    assert any("checksum" in error for error in report["integrity_errors"])
    assert any("Non-contiguous" in error for error in report["integrity_errors"])


def test_wrong_expected_source_frame_count_is_reported(tmp_path):
    report,*_=analyze_half(saved_half(tmp_path),200)
    assert not report["integrity_ok"]
    assert any("Expected 200" in error for error in report["integrity_errors"])


def test_stationary_run_resets_when_track_changes():
    values=np.zeros((100,10));values[:50,5]=1;values[50:,5]=2
    assert stationary_run(values,np.ones(100,dtype=bool))==(0,50)
    assert longest_run([False,True,True,False,True])==(1,3)
    assert longest_run([])==(0,0)


def test_trajectory_export_contains_units_and_retains_off_pitch_points(tmp_path):
    directory=saved_half(tmp_path)
    report,*_=analyze_half(directory)
    target=tmp_path/"trajectory.svg"
    trajectory_svg(report,target)
    text=target.read_text()
    assert "x (m)" in text and "y (m)" in text
    assert "y=70.0" in text


def test_offline_report_embeds_metrics_and_uses_relative_review_assets(tmp_path):
    half,*_=analyze_half(saved_half(tmp_path))
    report=dict(source="local <directory>",match_count=1,half_count=1,frames=half["frames"],
                detected=half["detected"],integrity_passed=1,generated_at="2026-11-04T00:00:00+00:00",halves=[half])
    target=tmp_path/"index.html"
    write_html_report(report,target)
    text=target.read_text()
    assert "const report=" in text and "contact_sheet.jpg" in text
    assert "local <directory>" not in text
    assert "local \\u003cdirectory>" in text
    assert "<script src=" not in text and "fetch(" not in text


def test_review_contact_sheet_and_clip_decode_without_rerunning_detector(tmp_path, monkeypatch):
    directory=saved_half(tmp_path,count=10)
    metadata,values,states,summary=analyze_half(directory)
    video=tmp_path/"source.mp4"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),25.,(640,480))
    assert writer.isOpened()
    for frame in range(10):
        writer.write(np.full((480,640,3),frame*10,np.uint8))
    writer.release()
    summary["video"]=str(video)
    import tactivision.review as review
    monkeypatch.setattr(review.shutil,"which",lambda command:None)
    output=tmp_path/"review"
    render_review(metadata,values,states,summary,output,clip_seconds=.08)
    assert cv2.imread(str(output/"contact_sheet.jpg")) is not None
    assert cv2.imread(str(output/"review_poster.jpg")) is not None
    capture=cv2.VideoCapture(str(output/"review.mp4"))
    try:
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT)==2
        assert capture.read()[0]
    finally:
        capture.release()
