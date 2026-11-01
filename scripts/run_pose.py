"""P0: YOLO pose + BoT-SORT on a webcam or video file — skeletons, track IDs, FPS.

    python scripts\\run_pose.py                      # webcam 0
    python scripts\\run_pose.py --source 1           # another webcam
    python scripts\\run_pose.py --source path\\to\\video.mp4 --save
Keys: q / Esc quit · space pause · b toggle boxes
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS  # noqa: E402
from edge.ingest.sources import open_source, resolve_model, source_name  # noqa: E402
from edge.perception.pose_tracker import PoseTracker, draw_people  # noqa: E402
from edge.viz import hud  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (try 960-1280 for crowded halls)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="cpu, 0, ... (default: auto)")
    ap.add_argument("--save", action="store_true", help="write annotated video to outputs/")
    ap.add_argument("--no-show", action="store_true", help="headless (use with --save)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    tracker = PoseTracker(resolve_model(args.model), args.device, args.imgsz, args.conf)
    cap = open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"model={args.model} device={tracker.device} imgsz={args.imgsz} source={args.source}")

    writer = None
    fps_ema, n, all_ids, show_boxes, t_start = None, 0, set(), True, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        people = tracker(frame)
        infer_ms = (time.perf_counter() - t0) * 1000
        draw_people(frame, people, draw_boxes=show_boxes)

        dt = time.perf_counter() - t0
        fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt
        all_ids.update(p.track_id for p in people if p.track_id is not None)
        hud(frame, [f"FPS {fps_ema:5.1f}  ({infer_ms:.0f} ms infer, {tracker.device})",
                    f"people {len(people)}  tracks seen {len(all_ids)}"])

        if args.save:
            if writer is None:
                OUTPUTS.mkdir(exist_ok=True)
                out = OUTPUTS / f"pose_{source_name(args.source)}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                                         (frame.shape[1], frame.shape[0]))
                print(f"saving -> {out}")
            writer.write(frame)

        n += 1
        if not args.no_show:
            cv2.imshow("SAGA P0 - pose + BoT-SORT", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                show_boxes = not show_boxes
            if key == ord(" "):
                cv2.waitKey(0)
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    print(f"{n} frames in {elapsed:.1f}s -> {n / max(elapsed, 1e-9):.1f} FPS overall, "
          f"{len(all_ids)} unique track IDs")


if __name__ == "__main__":
    main()
