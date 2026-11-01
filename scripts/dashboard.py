"""P3: invigilator dashboard without a camera — replay a recording, or a scripted demo class.

    python scripts\\dashboard.py --demo                  # 15-seat class with scripted behaviours (fast)
    python scripts\\dashboard.py --features outputs\\features_....csv [--seatmap outputs\\calib\\x.json]
Then open http://127.0.0.1:8000 . Live camera: run_seats.py --reason --dashboard.
"""
import argparse
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.app import serve_in_background  # noqa: E402
from api.live import LiveState, layout_from_positions, layout_from_seatmap  # noqa: E402
from config import OUTPUTS  # noqa: E402
from reasoner.reasoner import Reasoner  # noqa: E402
from reasoner.store import Store  # noqa: E402


def demo_session(calib_s: float, seed: int):
    """Synthetic 3x5 class with one of each behaviour after the baseline — for showing the dashboard.
    A demo, not evidence: evaluation lives in scripts/simulate.py."""
    from ml.injection import inject
    from ml.injection.session import synthetic_session
    sess = synthetic_session(rows=3, cols=5, duration_s=calib_s + 300, seed=seed)
    rng = np.random.default_rng(seed)
    t0 = calib_s + 15
    inject.copying(sess, rng, "B2", "B3", t0, k=6, window=90)            # B2 keeps looking at B3's desk
    inject.mutual(sess, rng, "A4", "A5", t0 + 40, k=3, window=60)        # A4 and A5 turn to each other
    inject.hands_under_desk(sess, rng, "C1", t0 + 70, hold=10)
    inject.turn_around(sess, rng, "A1", t0 + 110, k=2, window=40)
    return sess


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="scripted synthetic class")
    src.add_argument("--features", help="features CSV from run_seats.py")
    ap.add_argument("--seatmap", help="seat map JSON for the features CSV (layout + SAG positions)")
    ap.add_argument("--speed", type=float, default=None, help="playback speed (default: demo 4x, replay 1x)")
    ap.add_argument("--calib-seconds", type=float, default=None, help="baseline length (default demo 60 s)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    if args.demo:
        calib = args.calib_seconds or 60.0
        sess = demo_session(calib, args.seed)
        positions, layout, name = sess.pos, layout_from_positions(sess.pos), "Demo class (synthetic)"
        frames = sess.to_frames()
        speed = args.speed or 4.0
    else:
        from edge.calibration.seatmap import SeatMap
        from reasoner.reasoner import read_features_csv
        frames = read_features_csv(args.features)
        if not frames:
            sys.exit("No rows in the features file.")
        calib = args.calib_seconds
        speed = args.speed or 1.0
        name = f"Replay {Path(args.features).stem}"
        if args.seatmap:
            sm = SeatMap.load(args.seatmap)
            positions, layout = sm.positions(), layout_from_seatmap(sm)
        else:
            from ml.injection.session import _label_pos
            labels = sorted({r["seat"] for _, rows in frames for r in rows})
            positions = {s: _label_pos(s) for s in labels}
            layout = layout_from_positions(positions)

    reasoner = Reasoner(calib_s=calib, seat_positions=positions)
    store = Store(OUTPUTS / "saga.db")
    sid = store.new_session(name, args.features or "demo", reasoner.config)
    state = LiveState(name, args.features or "synthetic demo", "demo" if args.demo else "replay", store, sid)
    state.set_layout(layout)
    serve_in_background(state, port=args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard: {url}   (Ctrl+C to stop)  baseline {reasoner.baseline.calib_s:.0f}s, speed {speed}x")
    if not args.no_browser:
        webbrowser.open(url)

    wall0, t0 = time.perf_counter(), frames[0][0]
    try:
        for t, rows in frames:
            lag = (t - t0) / speed - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)
            out = reasoner.step(t, rows)
            for e in out.events:
                store.add_event(sid, e)
            for a in out.alerts:
                store.add_alert(sid, a)
                print(f"  {t:6.0f}s  {a.seat} {a.level.upper():<5} {'; '.join(a.reasons)}")
            if out.events or out.alerts:
                store.commit()
            state.update(t, reasoner, rows, out, fps=None, people=len(rows))
        print("Replay finished — the dashboard stays up. Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


if __name__ == "__main__":
    main()
