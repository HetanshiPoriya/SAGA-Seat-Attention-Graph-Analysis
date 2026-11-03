"""Core regression tests: run with  python -m pytest -q"""
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.calibration.seat_lock import SeatLocker  # noqa: E402
from edge.calibration.seatmap import SeatMap  # noqa: E402
from edge.perception.head_pose import LHIP, LSH, LWRIST, RHIP, RSH, RWRIST, head_pose  # noqa: E402
from edge.evidence.buffer import ClipBuffer  # noqa: E402
from edge.evidence.clips import Clip, ClipRecorder, purge  # noqa: E402
from edge.evidence.privacy import blur_faces, head_region  # noqa: E402
from edge.perception.interaction import object_features  # noqa: E402
from edge.perception.objects import ObjectDet, ObjectTracker, person_rois  # noqa: E402
from edge.perception.pose_tracker import Person  # noqa: E402
from ml.eval.injection_eval import run  # noqa: E402
from ml.injection import inject  # noqa: E402
from ml.injection.session import synthetic_session  # noqa: E402
from reasoner.baseline import CONTEXT_NAMES  # noqa: E402
from reasoner.rules.safe_expr import RuleSyntaxError, compile_condition  # noqa: E402
from reasoner.store import Store  # noqa: E402
from reasoner.transfer import TransferDetector  # noqa: E402


@pytest.mark.parametrize("expr", ["__import__('os')", "yaw.real > 1", "open('x')", "yaw[0] > 1",
                                  "unknown_name > 1", "(lambda: 1)()"])
def test_rule_expressions_are_sandboxed(expr):
    with pytest.raises(RuleSyntaxError):
        compile_condition(expr, CONTEXT_NAMES)


def test_rule_expression_nan_is_false():
    f = compile_condition("abs(yaw_z) > 2.5", CONTEXT_NAMES)
    assert f({"yaw_z": 3.0}) and not f({"yaw_z": math.nan})


def _face(yaw_deg, pitch_deg=0.0):
    pts3 = {0: (0, 0, 10), 1: (3.2, 0, 7), 2: (-3.2, 0, 7), 3: (7.5, 0, 0), 4: (-7.5, 0, 0)}
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    k, c = np.zeros((17, 2)), np.zeros(17)
    for i, (X, Y, Z) in pts3.items():
        x1, z1 = X * math.cos(y) + Z * math.sin(y), -X * math.sin(y) + Z * math.cos(y)
        k[i] = (100 + x1, 100 + Y * math.cos(p) - z1 * math.sin(p))
        c[i] = 0.9
    return k, c


@pytest.mark.parametrize("yaw", [-35, 0, 20, 50])
def test_head_yaw_recovered(yaw):
    assert abs(head_pose(*_face(yaw)).yaw - yaw) < 1.0


def test_back_of_head_is_away():
    k, c = _face(0)
    c[:3] = 0
    assert head_pose(k, c).facing == "away"


def test_seat_lock_absorbs_id_switch():
    sm = SeatMap.from_grid([[100, 400], [500, 420], [450, 150], [150, 140]], 3, 4, (640, 480))

    def person(tid, pt):
        k, c = np.zeros((17, 2)), np.zeros(17)
        k[11] = k[12] = pt
        c[11] = c[12] = 0.9
        return Person(tid, np.array([pt[0] - 20, pt[1] - 60, pt[0] + 20, pt[1] + 20.0]), 0.9, k, c)

    lock = SeatLocker(sm, 5, 5)
    a1 = sm.seat("A1").center
    for _ in range(10):
        lock.update([person(1, a1)])
    for _ in range(12):
        st = lock.update([person(7, a1)])
    assert st["A1"].track_id == 7 and lock.handovers == 1


def test_injected_copying_attributed_to_right_neighbour():
    sess = synthetic_session(rows=1, cols=4, duration_s=420, seed=2, distractors=False)
    rng = np.random.default_rng(0)
    gt = inject.copying(sess, rng, "A2", "A3", 250, k=4)
    res = run(sess, gt, 180)
    assert res.detected == [True]
    assert res.relational[0]["detected"] and not res.relational[0]["reversed"]


def test_untouched_clean_class_raises_no_alerts():
    sess = synthetic_session(rows=2, cols=4, duration_s=480, seed=5, distractors=False)
    assert run(sess, [], 180).alerts == []


# ------------------------------------------------------------------ P5: objects + interaction
def _student(cx=300.0, cy=200.0, sw=60.0, wrists=True):
    """A seated person facing the camera: shoulders, hips and (optionally) visible wrists."""
    k, c = np.zeros((17, 2)), np.zeros(17)
    k[LSH], k[RSH] = (cx - sw / 2, cy), (cx + sw / 2, cy)
    k[LHIP], k[RHIP] = (cx - sw / 2, cy + 1.6 * sw), (cx + sw / 2, cy + 1.6 * sw)
    c[[LSH, RSH, LHIP, RHIP]] = 0.9
    if wrists:
        k[LWRIST], k[RWRIST] = (cx - 0.7 * sw, cy + 0.9 * sw), (cx + 0.7 * sw, cy + 0.9 * sw)
        c[[LWRIST, RWRIST]] = 0.9
    return Person(1, np.array([cx - sw, cy - sw, cx + sw, cy + 2.4 * sw]), 0.9, k, c)


def _det(cls, cx, cy, size=14.0, conf=0.6, track_id=1, roi_kind="wrist"):
    return ObjectDet(cls, conf, np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2]),
                     track_id, roi_kind)


def test_hand_rois_cover_wrists_and_lap():
    p = _student()
    rois = person_rois(p, (600, 800))
    kinds = [k for k, _ in rois]
    assert kinds.count("lap") == 1 and len(rois) >= 2      # two wrists (or merged) + the lap
    for _, b in rois:
        assert 0 <= b[0] < b[2] <= 799 and 0 <= b[1] < b[3] <= 599


def test_object_near_wrist_counts_as_interaction():
    p = _student()
    near = object_features(p, [_det("phone", 300 - 0.7 * 60, 200 + 0.9 * 60)])
    assert near["obj_class"] == "phone" and near["obj_interact"] == 1.0 and near["obj_dist"] < 0.3
    far = object_features(p, [_det("phone", 300 + 400, 200)])
    assert far["obj_interact"] == 0.0 and far["obj_dist"] > 1.0


def test_object_below_hips_is_flagged_as_lap():
    p = _student()
    f = object_features(p, [_det("phone", 300, 200 + 2.0 * 60)])
    assert f["obj_below"] == 1.0


def test_prohibited_object_wins_over_allowed_one():
    p = _student()
    f = object_features(p, [_det("bottle", 300 - 0.7 * 60, 200 + 0.9 * 60),
                            _det("phone", 300 + 0.7 * 60, 200 + 0.9 * 60)])
    assert f["obj_class"] == "phone"


def test_object_tracker_keeps_one_id_and_separates_classes():
    tr = ObjectTracker()
    a = tr.update(0.0, [_det("phone", 100, 100, track_id=None)], scale=60)[0]
    b = tr.update(0.3, [_det("phone", 104, 103, track_id=None)], scale=60)[0]
    c = tr.update(0.6, [_det("chit", 104, 103, track_id=None)], scale=60)[0]
    assert a.track_id == b.track_id and c.track_id != a.track_id


def _rows(seat, t, oid=7, cls="chit"):
    return {"seat": seat, "obj_id": float(oid), "obj_class": cls, "obj_interact": 1.0, "obj_conf": 0.7}


def test_transfer_between_neighbours_names_both_seats():
    td = TransferDetector({"A1": (1.0, 0.0), "A2": (2.0, 0.0), "A3": (3.0, 0.0)})
    for t in np.arange(0, 1.6, 0.1):      # A1 holds the chit for 1.5 s
        assert td.observe(float(t), [_rows("A1", t)]) == []
    out = td.observe(2.0, [_rows("A2", 2.0)])
    assert {s for s, _ in out} == {"A1", "A2"}
    ev = dict(out)["A1"]
    assert ev.rule_id == "CHIT_TRANSFER" and ev.related == ["A2"] and "A1 to A2" in ev.message


def test_slow_handover_is_not_a_transfer():
    td = TransferDetector({"A1": (1.0, 0.0), "A2": (2.0, 0.0)})
    for t in np.arange(0, 1.6, 0.1):
        td.observe(float(t), [_rows("A1", t)])
    assert td.observe(30.0, [_rows("A2", 30.0)]) == []    # 28 s later: two unrelated sightings


def test_injected_phone_and_chit_pass_are_detected():
    sess = synthetic_session(rows=1, cols=3, duration_s=320, seed=1, distractors=False)
    rng = np.random.default_rng(0)
    lap = run(sess, inject.phone_in_lap(sess, rng, "A1", 250.0), 180)
    assert lap.detected == [True]
    sess2 = synthetic_session(rows=1, cols=3, duration_s=320, seed=1, distractors=False)
    res = run(sess2, inject.chit_pass(sess2, rng, "A1", "A2", 250.0), 180)
    assert res.detected == [True, True]
    assert all(r["detected"] and not r["wrong_target"] for r in res.relational)


def test_allowed_object_in_the_hand_raises_nothing():
    sess = synthetic_session(rows=1, cols=3, duration_s=320, seed=1, distractors=False)
    inject._hold_object(sess, "A1", 250.0, 20.0, "bottle", 55, below=False)
    assert [e for e in run(sess, [], 180).events if e.rule_id.startswith("OBJECT")] == []


# ------------------------------------------------------------------ P6: evidence clips + privacy
def _frame_with_face(w=320, h=240, cx=160, cy=80, r=28):
    """A frame with a high-detail 'face' patch, so masking is measurable."""
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), 40, np.uint8)
    img[cy - r:cy + r, cx - r:cx + r] = rng.integers(0, 255, (2 * r, 2 * r, 3), dtype=np.uint8)
    return img


def _person_at(cx=160, cy=80, sw=60):
    k, c = np.zeros((17, 2)), np.zeros(17)
    k[0], k[1], k[2] = (cx, cy), (cx - 8, cy - 6), (cx + 8, cy - 6)      # nose, eyes
    k[3], k[4] = (cx - 18, cy - 2), (cx + 18, cy - 2)                    # ears
    k[5], k[6] = (cx - sw / 2, cy + 50), (cx + sw / 2, cy + 50)          # shoulders
    c[[0, 1, 2, 3, 4, 5, 6]] = 0.9
    return Person(1, np.array([cx - sw, cy - 40, cx + sw, cy + 140.0]), 0.9, k, c)


def _detail(img, box):
    x1, y1, x2, y2 = box
    return float(np.std(img[y1:y2, x1:x2].astype(float)))


def test_face_region_found_from_keypoints_and_from_box():
    p = _person_at()
    box = head_region(p)
    assert box is not None and box[0] < 160 < box[2] and box[1] < 80 < box[3]
    blind = _person_at()
    blind.kpt_conf[:5] = 0.0                      # head not visible: fall back to the top of the box
    top = head_region(blind)
    assert top is not None and top[3] - top[1] < (blind.box[3] - blind.box[1])


def _distinct_colours(img, box):
    x1, y1, x2, y2 = box
    patch = img[y1:y2, x1:x2].reshape(-1, 3)
    return len(np.unique(patch, axis=0))


def test_masking_leaves_only_flat_blocks_where_the_face_was():
    """The real privacy property: the pixels become block-constant, so the face information is
    gone. (Standard deviation is the WRONG check — a mosaic keeps coarse hair/background contrast
    and still measures 'detailed'; verified against real footage, where the masked centre of a head
    holds ~16 distinct colours.)"""
    img = _frame_with_face()
    centre = (150, 70, 170, 90)
    assert _distinct_colours(img, centre) > 100       # busy to start with
    assert blur_faces(img, [_person_at()]) == 1
    # the 20 px probe spans at most 4x4 mosaic blocks, so a few dozen colours at the very most
    assert _distinct_colours(img, centre) <= 25       # down from 400: the face is flat blocks
    assert _detail(img, (0, 200, 60, 240)) < 1.0      # the rest of the frame is untouched


def test_every_person_is_masked_not_only_the_alerted_one():
    img = _frame_with_face()
    img[70:90, 250:290] = 200
    assert blur_faces(img, [_person_at(), _person_at(cx=270, cy=80)]) == 2


def test_buffer_keeps_only_the_window():
    buf = ClipBuffer(seconds=2.0, max_width=160)
    for i in range(40):
        buf.add(i * 0.1, _frame_with_face(), [_person_at()])
    first, last = buf.covered
    assert last - first <= 2.0 + 1e-6 and buf.memory_mb < 1.0
    assert len(buf.window(3.0, 3.5)) >= 3


def test_recorder_waits_for_the_post_roll_then_writes_a_clip(tmp_path):
    buf = ClipBuffer(seconds=30.0, max_width=160)
    rec = ClipRecorder(buf, tmp_path, pre_s=0.5, post_s=0.5, fps=10)
    for i in range(10):                            # 0.0 .. 0.9 s
        buf.add(i * 0.1, _frame_with_face(), [_person_at()])
    assert rec.request(0.5, "A1", alert_id="alert-1")
    assert rec.pump(0.5) == []                     # post-roll has not happened yet
    for i in range(10, 20):
        buf.add(i * 0.1, _frame_with_face(), [_person_at()])
    clips = rec.pump(1.9)
    assert len(clips) == 1
    clip = clips[0]
    assert clip.path.exists() and clip.frames > 5 and clip.faces_masked == clip.frames
    assert clip.alert_id == "alert-1" and clip.seat == "A1"


def test_clip_frames_on_disk_have_masked_faces(tmp_path):
    buf = ClipBuffer(seconds=30.0, max_width=320)
    rec = ClipRecorder(buf, tmp_path, pre_s=0.5, post_s=0.2, fps=10, skeleton=False)
    for i in range(12):
        buf.add(i * 0.1, _frame_with_face(), [_person_at()])
    rec.request(0.4, "B2")
    clip = rec.pump(1.2)[0]
    cap = cv2.VideoCapture(str(clip.path))
    ok, written = cap.read()
    cap.release()
    assert ok
    assert _detail(written, (140, 60, 180, 100)) < _detail(_frame_with_face(), (140, 60, 180, 100)) * 0.6


def test_retention_deletes_only_old_clips(tmp_path):
    old, new = tmp_path / "clip_A1_old.mp4", tmp_path / "clip_A1_new.mp4"
    old.write_bytes(b"x"), new.write_bytes(b"y")
    os.utime(old, (time.time() - 40 * 86400,) * 2)
    gone = purge(tmp_path, retention_days=30)
    assert gone == [old] and not old.exists() and new.exists()


def test_clip_is_recorded_and_every_view_is_audited(tmp_path):
    store = Store(tmp_path / "t.db")
    sid = store.new_session("t", "test")
    clip = Clip("c1", tmp_path / "clip_A1_c1.mp4", "alert-1", "A1", 0.0, 8.0, 80, 80)
    store.add_clip(sid, clip)
    assert store.clip_for_alert("alert-1")["id"] == "c1"
    store.log_clip_view("c1", "invigilator-1")
    store.log_clip_view("c1", "invigilator-2")
    assert [v["viewer"] for v in store.clip_views("c1")] == ["invigilator-1", "invigilator-2"]
    store.mark_clips_deleted([str(clip.path)])
    assert store.clip_for_alert("alert-1") is None      # deleted clips are not offered any more
    store.close()


def test_dashboard_serves_the_clip_and_logs_every_view(tmp_path):
    """The clip endpoint is the only way a face-bearing file leaves the machine, so it must exist,
    404 cleanly, and write an audit row each time (spec §13 'audit log on every clip view')."""
    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.live import LiveState

    store = Store(tmp_path / "t.db")
    sid = store.new_session("t", "test")
    path = tmp_path / "clip_A1_c1.mp4"
    path.write_bytes(b"not really mp4, but a real file")
    store.add_clip(sid, Clip("c1", path, "alert-1", "A1", 0.0, 8.0, 80, 80))

    state = LiveState("t", "test", "test", store, sid)
    client = TestClient(create_app(state))
    assert client.get("/api/clips/nope").status_code == 404
    r = client.get("/api/clips/c1")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    client.get("/api/clips/c1")
    assert len(store.clip_views("c1")) == 2            # both views audited
    assert client.get("/api/clips/c1/views").json()["views"][0]["viewer"]

    store.mark_clips_deleted([str(path)])
    assert client.get("/api/clips/c1").status_code == 404   # retention removed it: no longer served
    store.close()
