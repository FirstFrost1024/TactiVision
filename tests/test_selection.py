import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tactivision.calibration import Calibration
from tactivision.candidate_cache import CachedBallDetector, CandidateWriter
from tactivision.detector import Detection
from tactivision.pipeline import reconstruct
from tactivision.region import PitchRegion
from tactivision.tracking import BallAssociator, Candidate


def candidate(pixel, world, confidence=.8):
    u,v=pixel
    return Candidate(Detection((u-1,v-1,u+1,v+1),confidence),np.array(world,dtype=float))


@pytest.fixture
def region():
    return PitchRegion([[0,0],[100,0],[100,100],[0,100]])


def test_region_uses_curved_boundary_instead_of_convex_hull(tmp_path):
    path=tmp_path/'keypoints.json'
    path.write_text(json.dumps({'(0,0)':[0,0],'(105,0)':[100,0],
                               '(105,68)':[100,100],'(52.5,68)':[50,70],'(0,68)':[0,100]}))
    calibration=SimpleNamespace(pitch_length=105,pitch_width=68,undistort=lambda points:points+[10,20])
    region=PitchRegion.load(path,calibration)
    assert region.distance((50,50))>0
    assert region.distance((50,90))<0
    undistorted=PitchRegion.load(path,calibration,'undistorted')
    assert undistorted.distance((60,70))>0
    assert undistorted.distance((60,110))<0


def test_outside_spare_ball_cannot_start_a_track_even_with_higher_confidence(region):
    tracker=BallAssociator(region=region)
    game=candidate((50,50),(50,30),.2)
    spare=candidate((50,110),(50,70),.99)
    assert tracker.select([spare,game],0) is game
    assert tracker.select([spare],1) is None


def test_no_inside_detection_stays_missing_instead_of_selecting_a_spare(region):
    tracker=BallAssociator(region=region)
    assert tracker.select([candidate((50,110),(50,70),.99)],0) is None
    assert tracker.track_id==0


def test_known_ball_can_cross_touchline_temporarily_without_clamping(region):
    tracker=BallAssociator(region=region,static_seconds=.05)
    tracker.select([candidate((99,50),(104,30))],0)
    outside=candidate((102,50),(106,30))
    assert tracker.select([outside],.04) is outside
    assert tracker.select([outside],.08) is outside
    assert outside.pitch_xy[0]>105
    assert tracker.select([outside],.12) is None
    assert tracker.excluded_static==1


def test_stationary_ball_inside_field_is_not_suppressed(region):
    tracker=BallAssociator(region=region,static_seconds=.1)
    ball=candidate((50,50),(50,30))
    for i in range(50):
        assert tracker.select([ball],i*.04) is ball
    assert tracker.excluded_static==0


def cache_fixture(tmp_path):
    video=tmp_path/'video.mp4';video.write_bytes(b'controlled source identity')
    stat=video.stat();directory=tmp_path/'cache';directory.mkdir()
    writer=CandidateWriter(directory,dict(video=str(video.resolve()),video_size=stat.st_size,
                           video_mtime_ns=stat.st_mtime_ns,start_frame=10,class_id=32,
                           provenance={'confidence':.15},image_space='distorted'))
    return directory,video,writer


def test_cache_preserves_all_candidates_and_can_seek_and_filter(tmp_path):
    directory,video,writer=cache_fixture(tmp_path)
    writer.write(10,[])
    writer.write(11,[Detection((1,2,3,4),.2),Detection((10,20,30,40),.9)])
    writer.finish()
    reader=CachedBallDetector(directory,video,start_frame=11,confidence=.3)
    try:
        result=reader.detect(None)
        assert len(result)==1 and result[0].xyxy==(10,20,30,40)
        assert reader.class_id==32
    finally:
        reader.close()
    with pytest.raises(ValueError,match='below'):
        CachedBallDetector(directory,video,start_frame=10,confidence=.1)


def test_unfinished_or_changed_cache_is_not_reused(tmp_path):
    directory,video,writer=cache_fixture(tmp_path)
    writer.write(10,[]);writer.close()
    assert not (directory/'candidates_meta.json').exists()
    with pytest.raises(FileNotFoundError):
        CachedBallDetector(directory,video,start_frame=10)
    writer=CandidateWriter(directory,writer.metadata);writer.write(10,[]);writer.finish()
    with (directory/'candidates.jsonl.gz').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        CachedBallDetector(directory,video,start_frame=10)


def test_cache_is_not_published_before_requested_frames_are_written(tmp_path):
    directory,video,writer=cache_fixture(tmp_path)
    writer.metadata['requested_frames']=2
    writer.write(10,[])
    with pytest.raises(ValueError,match='Incomplete'):
        writer.finish()
    assert not (directory/'candidates.jsonl.gz').exists()
    assert not (directory/'candidates_meta.json').exists()


def test_cached_reassociation_matches_live_selection_without_detector_inference(tmp_path):
    video=tmp_path/'source.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),25.,(640,480))
    assert writer.isOpened()
    for _ in range(5):writer.write(np.zeros((480,640,3),dtype=np.uint8))
    writer.release()
    K=np.array([[300.,0,320],[0,300.,240],[0,0,1]])
    H=np.array([[4.,0,100],[0,3.,100],[0,0,1.]])
    calibration=Calibration(K,np.zeros(4),K,H)
    pixels=calibration.ground_to_image([[20,30],[80,65]])
    detections=[Detection((u-1,v-1,u+1,v+1),c) for (u,v),c in zip(pixels,[.3,.99])]
    u,v=pixels[0];region=PitchRegion([[u-20,v-20],[u+20,v-20],[u+20,v+20],[u-20,v+20]])
    detector=SimpleNamespace(detect=lambda frame:detections,class_id=32)
    live=tmp_path/'live';replay=tmp_path/'replay'
    reconstruct(video,calibration,detector,live,pitch_region=region,save_candidates=True,
                provenance={'confidence':.15})
    cached=CachedBallDetector(live,video)
    try:
        reconstruct(video,calibration,cached,replay,pitch_region=region,max_frames=cached.frames)
    finally:
        cached.close()
    with (live/'ball_positions.csv').open() as first,(replay/'ball_positions.csv').open() as second:
        a,b=list(csv.DictReader(first)),list(csv.DictReader(second))
    assert a==b
    assert all(float(row['x_m'])==pytest.approx(20,abs=1e-6) for row in a)
    assert all(row['candidate_count']=='2' and row['playing_candidate_count']=='1' for row in a)
