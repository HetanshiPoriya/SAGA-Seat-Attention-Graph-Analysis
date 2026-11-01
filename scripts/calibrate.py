"""P1: one-time hall calibration (spec §6.2) -> seat map JSON.

Manual grid (click the 4 corners of the desk area):
    python scripts\\calibrate.py --source 0 --rows 2 --cols 3
    python scripts\\calibrate.py --source clip.mp4 --frame 100 --rows 5 --cols 6
  Click order: FRONT-LEFT, FRONT-RIGHT, BACK-RIGHT, BACK-LEFT (front row = row A, nearest the
  invigilator/board). Keys: w/s rows +/-  a/d cols +/-  u undo  r reset  Enter/space save  q quit

Automatic pseudo seat map (no clicking; one seat per student who sits still):
    python scripts\\calibrate.py --source clip.mp4 --auto 60
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS  # noqa: E402
from edge.calibration.seatmap import AutoSeatBuilder, SeatMap  # noqa: E402
from edge.ingest.sources import source_name  # noqa: E402

CORNER_NAMES = ["FRONT-LEFT", "FRONT-RIGHT", "BACK-RIGHT", "BACK-LEFT"]
MAX_W = 1280


def grab_frame(src: str, index: int):
    cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if src.isdigit() else cv2.VideoCapture(src)
    if src.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        index = max(index, 15)  # let webcam exposure settle
    frame = None
    for _ in range(index + 1):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    if frame is None:
        sys.exit(f"Could not read a frame from {src}")
    return frame


def default_out(src: str) -> Path:
    return OUTPUTS / "calib" / f"{source_name(src)}.json"


def manual(frame, rows, cols, out: Path):
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_W / w)
    pts: list[list[float]] = []
    win = "SAGA calibration"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x / scale, y / scale])

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = frame.copy()
        sm = SeatMap.from_grid(pts, rows, cols, (w, h)) if len(pts) == 4 else None
        if sm:
            sm.draw(view)
        for i, p in enumerate(pts):
            cv2.circle(view, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
            cv2.putText(view, CORNER_NAMES[i], (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        msg = (f"Click {CORNER_NAMES[len(pts)]} corner" if len(pts) < 4 else "Enter/space = save")
        info = [msg, f"rows {rows} (w/s)  cols {cols} (a/d)  u undo  r reset  q quit"]
        lh = int(28 / scale)  # sizes in full-res pixels so the bar looks the same after resizing
        for i, t in enumerate(info):
            cv2.rectangle(view, (0, lh * i), (int(620 / scale), lh * (i + 1)), (0, 0, 0), -1)
            cv2.putText(view, t, (8, int(lh * (i + 0.72))), cv2.FONT_HERSHEY_SIMPLEX, 0.6 / scale,
                        (255, 255, 255), max(1, int(1 / scale)), cv2.LINE_AA)
        cv2.imshow(win, cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            cv2.destroyAllWindows()
            sys.exit("Cancelled — nothing saved.")
        if k == ord("u") and pts:
            pts.pop()
        elif k == ord("r"):
            pts.clear()
        elif k == ord("w"):
            rows += 1
        elif k == ord("s") and rows > 1:
            rows -= 1
        elif k == ord("d"):
            cols += 1
        elif k == ord("a") and cols > 1:
            cols -= 1
        elif k in (13, 32) and sm:
            cv2.destroyAllWindows()
            return sm, view


def auto(src: str, n_frames: int, model: str, imgsz: int):
    from edge.perception.pose_tracker import PoseTracker
    from edge.ingest.sources import open_source, resolve_model
    tracker = PoseTracker(resolve_model(model), imgsz=imgsz)
    cap = open_source(src)
    builder, frame = AutoSeatBuilder(), None
    for i in range(n_frames):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
        builder.add(tracker(f))
        print(f"\rcollecting seats: frame {i + 1}/{n_frames}", end="")
    print()
    cap.release()
    if frame is None:
        sys.exit("No frames read.")
    sm = builder.build((frame.shape[1], frame.shape[0]))
    return sm, sm.draw(frame.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--frame", type=int, default=0, help="video frame index to calibrate on")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--auto", type=int, default=0, metavar="N", help="auto seat map from first N frames")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else default_out(args.source)
    if args.auto:
        sm, preview = auto(args.source, args.auto, args.model, args.imgsz)
    else:
        sm, preview = manual(grab_frame(args.source, args.frame), args.rows, args.cols, out)
    sm.save(out)
    cv2.imwrite(str(out.with_suffix(".jpg")), preview)
    print(f"{sm.mode} seat map: {len(sm.seats)} seats ({sm.rows} rows x {sm.cols} cols)")
    print(f"saved {out}\npreview {out.with_suffix('.jpg')}")


if __name__ == "__main__":
    main()
