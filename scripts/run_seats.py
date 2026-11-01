"""P1/P2: pose + tracking + seat locking + per-seat head pose (+ live reasoning), keypoints-only log.

    python scripts\\run_seats.py --source 0 --calib outputs\\calib\\webcam.json
    python scripts\\run_seats.py --source clip.mp4 --auto-seats 60 --save
    python scripts\\run_seats.py --source 0 --auto-seats 30 --reason --calib-seconds 30   # P2 live
Keys: q / Esc quit · space pause
Writes outputs\\features_<source>_<time>.csv (one row per seat per frame; no images, no faces);
with --reason also events/alerts to outputs\\saga.db.
"""
import argparse
import csv
import math
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS  # noqa: E402
from edge.calibration.seat_lock import SeatLocker  # noqa: E402
from edge.calibration.seatmap import AutoSeatBuilder, SeatMap  # noqa: E402
from edge.ingest.sources import open_source, resolve_model, source_name  # noqa: E402
from edge.perception.head_pose import body_features, hand_features, head_pose  # noqa: E402
from edge.perception.pose_tracker import PoseTracker, id_color  # noqa: E402
from edge.viz import draw_head_arrow, draw_sag_edges, draw_skeleton, hud, label, ui_scale  # noqa: E402

CSV_FIELDS = ["frame", "t", "seat", "track_id", "facing", "yaw", "pitch", "yaw_s", "pitch_s",
              "lean_x", "shoulder_tilt", "shoulder_ratio", "wrist_drop", "wrists_vis", "hand_face",
              "head_px", "kp_conf"]
CSV_FMT = {"yaw": ".1f", "pitch": ".1f", "yaw_s": ".1f", "pitch_s": ".1f", "lean_x": ".3f",
           "shoulder_tilt": ".1f", "shoulder_ratio": ".3f", "wrist_drop": ".3f", "wrists_vis": ".0f",
           "hand_face": ".3f", "head_px": ".0f", "kp_conf": ".2f"}
LEVEL_COLORS = {"green": (60, 200, 60), "amber": (0, 165, 255), "red": (0, 0, 255), "grey": (128, 128, 128),
                "calibrating": (200, 200, 200)}


class Smoother:
    """Per-seat EMA of yaw/pitch; resets when a seat has been empty for a while."""

    def __init__(self, alpha=0.4, reset_after=15):
        self.alpha, self.reset_after = alpha, reset_after
        self.state: dict[str, tuple[float, float, int]] = {}

    def __call__(self, seat, yaw, pitch, frame):
        if yaw is None:
            return None, None
        prev = self.state.get(seat)
        if prev is None or frame - prev[2] > self.reset_after:
            y, p = yaw, pitch
        else:
            y = prev[0] + self.alpha * (yaw - prev[0])
            p = pitch if pitch is None or prev[1] is None else prev[1] + self.alpha * (pitch - prev[1])
        self.state[seat] = (y, p, frame)
        return y, p


def fmt(v, spec="+.0f"):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def draw_feed(frame, lines):
    """Latest events/alerts along the bottom of the frame."""
    if not lines:
        return
    h, w = frame.shape[:2]
    k = ui_scale(frame)
    lh = int(22 * k)
    top = h - lh * len(lines) - int(10 * k)
    cv2.rectangle(frame, (0, top), (w, h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        color = (0, 0, 255) if "RED" in text else (0, 165, 255) if "AMBER" in text else (255, 255, 255)
        cv2.putText(frame, text[:110], (int(8 * k), top + int(20 * k) + lh * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * k, color, max(1, int(k)), cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calib", help="seat map JSON from calibrate.py")
    g.add_argument("--auto-seats", type=int, metavar="N", help="build a pseudo seat map from the first N frames")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--lock-frames", type=int, default=8)
    ap.add_argument("--save", action="store_true", help="write annotated video to outputs/")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--reason", action="store_true", help="P2: baseline + rules + risk, live")
    ap.add_argument("--calib-seconds", type=float, default=None,
                    help="baseline length (default from reasoner/config.yaml = 180; ~30 for a webcam demo)")
    ap.add_argument("--db", default=str(OUTPUTS / "saga.db"))
    ap.add_argument("--dashboard", action="store_true", help="P3: serve the invigilator dashboard (implies --reason)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    args.reason = args.reason or args.dashboard

    tracker = PoseTracker(resolve_model(args.model), imgsz=args.imgsz, conf=args.conf)
    cap = open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    live = args.source.isdigit()  # webcam: real time (processing fps < camera fps); file: frame time
    stamp = f"{source_name(args.source)}_{time.strftime('%Y%m%d_%H%M%S')}"
    OUTPUTS.mkdir(exist_ok=True)

    reasoner = store = session_id = live_state = None
    feed: list[str] = []  # latest events/alerts shown on screen

    def start_reasoner(sm: SeatMap):
        """Created once seats are known: the Seat Attention Graph needs their positions."""
        nonlocal reasoner, store, session_id, live_state
        from reasoner.reasoner import Reasoner
        from reasoner.store import Store
        reasoner = Reasoner(calib_s=args.calib_seconds, seat_positions=sm.positions())
        store = Store(args.db)
        session_id = store.new_session(stamp, args.source, reasoner.config)
        if args.dashboard:
            from api.app import serve_in_background
            from api.live import LiveState, layout_from_seatmap
            live_state = LiveState(stamp, args.source, "live", store, session_id)
            live_state.set_layout(layout_from_seatmap(sm))
            serve_in_background(live_state, port=args.port)
            print(f"Dashboard: http://127.0.0.1:{args.port}")

    seatmap = SeatMap.load(args.calib) if args.calib else None
    builder = None if seatmap else AutoSeatBuilder()
    locker = SeatLocker(seatmap, args.lock_frames, args.lock_frames) if seatmap else None
    if seatmap and args.reason:
        start_reasoner(seatmap)
    smooth = Smoother()

    csv_file = None if args.no_csv else open(OUTPUTS / f"features_{stamp}.csv", "w", newline="")
    log = csv.DictWriter(csv_file, CSV_FIELDS) if csv_file else None
    if log:
        log.writeheader()

    writer, fps_ema, n, t_start = None, None, 0, time.perf_counter()
    present = defaultdict(int)
    yaws = defaultdict(list)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        t_now = (t0 - t_start) if live else n / src_fps
        people = tracker(frame)

        if seatmap is None:  # still collecting the auto seat map
            builder.add(people)
            for p in people:
                draw_skeleton(frame, p, id_color(p.track_id))
            status = f"building seat map {builder.frames}/{args.auto_seats}"
            if builder.frames >= args.auto_seats:
                try:
                    seatmap = builder.build((frame.shape[1], frame.shape[0]))
                except ValueError:  # nobody in view long enough yet: start collecting again
                    print("No seated person found yet - make sure you are in view of the camera. Retrying...")
                    builder = AutoSeatBuilder()
                    status = "no person found - sit in view of the camera"
            if seatmap is not None:
                seatmap.save(OUTPUTS / "calib" / f"{stamp}_auto.json")
                locker = SeatLocker(seatmap, args.lock_frames, args.lock_frames)
                if args.reason:
                    start_reasoner(seatmap)
                print(f"auto seat map: {len(seatmap.seats)} seats in {seatmap.rows} rows "
                      f"-> {OUTPUTS / 'calib' / f'{stamp}_auto.json'}")
        else:
            seats = locker.update(people)
            if reasoner:
                views = {lbl: reasoner.seat_view(lbl) for lbl in seats}
                seatmap.draw(frame, locker.occupied, colors={
                    lbl: LEVEL_COLORS["calibrating" if v.calibrating else v.level] for lbl, v in views.items()})
            else:
                seatmap.draw(frame, locker.occupied)
            seated, rows = set(), []
            for lbl, st in seats.items():
                p = st.person
                if p is None:
                    continue
                seated.add(id(p))
                hp = head_pose(p.kpts, p.kpt_conf)
                ys, ps = smooth(lbl, hp.yaw, hp.pitch, n)
                present[lbl] += 1
                if ys is not None:
                    yaws[lbl].append(ys)
                nan = float("nan")
                row = {"seat": lbl, "facing": hp.facing,
                       "yaw": nan if hp.yaw is None else hp.yaw, "pitch": nan if hp.pitch is None else hp.pitch,
                       "yaw_s": nan if ys is None else ys, "pitch_s": nan if ps is None else ps,
                       "head_px": hp.head_px, "kp_conf": float(np.mean(p.kpt_conf[:7]))}
                row.update({k: v for k, v in body_features(p.kpts, p.kpt_conf, p.box).items() if k in CSV_FMT})
                row.update(hand_features(p.kpts, p.kpt_conf))
                rows.append(row)

                color = id_color(zlib.crc32(lbl.encode()))
                draw_skeleton(frame, p, color)
                draw_head_arrow(frame, hp.origin, ys, ps, hp.head_px)
                head = "away" if hp.facing == "away" else f"y{fmt(ys)} p{fmt(ps)}"
                text = f"{lbl}  {head}"
                if reasoner:
                    v = views[lbl]
                    text += "  calibrating" if v.calibrating else f"  risk {v.risk:.2f}"
                    color = LEVEL_COLORS["calibrating" if v.calibrating else v.level]
                label(frame, text, (p.box[0], p.box[1]), color)
                if log:
                    log.writerow({"frame": n, "t": round(t_now, 3), "track_id": p.track_id,
                                  **{k: (fmt(v, CSV_FMT[k]) if k in CSV_FMT else v)
                                     for k, v in row.items() if k in CSV_FIELDS}})
            for p in people:  # detected but not (yet) locked to a seat
                if id(p) not in seated:
                    draw_skeleton(frame, p, (150, 150, 150))
                    label(frame, "?", (p.box[0], p.box[1]), (150, 150, 150))
            status = (f"seats occupied {len(locker.occupied)}/{len(seatmap.seats)}  "
                      f"people {len(people)}  ID switches absorbed {locker.handovers}")

            if reasoner:
                out = reasoner.step(t_now, rows)
                if live_state:
                    live_state.update(t_now, reasoner, rows, out, fps=fps_ema, people=len(people))
                for e in out.events:
                    store.add_event(session_id, e)
                    feed.append(f"[{t_now:5.0f}s] {e.seat}: {e.message}")
                    print(f"EVENT  t={t_now:.0f}s {e.seat} {e.rule_id}: {e.message}")
                for a in out.alerts:
                    store.add_alert(session_id, a)
                    feed.append(f"[{t_now:5.0f}s] {a.seat} -> {a.level.upper()} (risk {a.risk:.2f})")
                    print(f"ALERT  t={t_now:.0f}s {a.seat} {a.level.upper()} risk={a.risk:.2f} reasons={a.reasons}")
                if out.events or out.alerts:
                    store.commit()
                left = (reasoner.baseline.calib_s - (t_now - reasoner.baseline.start_t)
                        if reasoner.baseline.start_t is not None else reasoner.baseline.calib_s)
                status += f"  |  {'baseline: sit normally ' + str(int(left)) + 's' if left > 0 else 'monitoring'}"
                feed[:] = feed[-4:]
                draw_sag_edges(frame, {x.label: x.center for x in seatmap.seats}, reasoner.edges(t_now, 120))
                draw_feed(frame, feed)

        dt = time.perf_counter() - t0
        fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt
        hud(frame, [f"FPS {fps_ema:5.1f} ({tracker.device})", status])

        if args.save:
            if writer is None:
                out = OUTPUTS / f"seats_{stamp}.mp4"
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                                         (frame.shape[1], frame.shape[0]))
                print(f"saving -> {out}")
            writer.write(frame)
        n += 1
        if not args.no_show:
            cv2.imshow("SAGA - seats", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                cv2.waitKey(0)
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()
        print(f"features -> {csv_file.name}")
    if store:
        store.close()
        print(f"events/alerts -> {args.db} (session {session_id[:8]})")
    cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(f"\n{n} frames in {elapsed:.1f}s -> {n / max(elapsed, 1e-9):.1f} FPS overall")
    if locker:
        all_tracks = {t for st in locker.seats.values() for t in st.track_history}
        print(f"seat handovers (tracker ID switches absorbed by seat locking): {locker.handovers}")
        print(f"{'seat':<6}{'present':>9}{'yaw mean':>10}{'yaw sd':>8}  track IDs")
        for lbl, st in locker.seats.items():
            if present[lbl]:
                y = np.asarray(yaws[lbl]) if yaws[lbl] else np.array([np.nan])
                print(f"{lbl:<6}{present[lbl] / max(locker.frame + 1, 1):>8.0%}{np.nanmean(y):>10.1f}"
                      f"{np.nanstd(y):>8.1f}  {st.track_history}")
        print(f"{len(all_tracks)} track IDs mapped onto {sum(1 for l in present if present[l])} seats")


if __name__ == "__main__":
    main()
