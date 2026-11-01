"""Core regression tests: run with  python -m pytest -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.calibration.seat_lock import SeatLocker  # noqa: E402
from edge.calibration.seatmap import SeatMap  # noqa: E402
from edge.perception.head_pose import head_pose  # noqa: E402
from edge.perception.pose_tracker import Person  # noqa: E402
from ml.eval.injection_eval import run  # noqa: E402
from ml.injection import inject  # noqa: E402
from ml.injection.session import synthetic_session  # noqa: E402
from reasoner.baseline import CONTEXT_NAMES  # noqa: E402
from reasoner.rules.safe_expr import RuleSyntaxError, compile_condition  # noqa: E402


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
