"""Core regression tests: run with  python -m pytest -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.calibration.seat_lock import SeatLocker  # noqa: E402
from edge.calibration.seatmap import SeatMap  # noqa: E402
from edge.perception.head_pose import LHIP, LSH, LWRIST, RHIP, RSH, RWRIST, head_pose  # noqa: E402
from edge.perception.interaction import object_features  # noqa: E402
from edge.perception.objects import ObjectDet, ObjectTracker, person_rois  # noqa: E402
from edge.perception.pose_tracker import Person  # noqa: E402
from ml.eval.injection_eval import run  # noqa: E402
from ml.injection import inject  # noqa: E402
from ml.injection.session import synthetic_session  # noqa: E402
from reasoner.baseline import CONTEXT_NAMES  # noqa: E402
from reasoner.rules.safe_expr import RuleSyntaxError, compile_condition  # noqa: E402
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
