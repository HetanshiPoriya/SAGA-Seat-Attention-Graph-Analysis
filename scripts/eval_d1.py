"""P1: evaluate head-pose / posture features on D1 (needs the labelled dataset).

    python scripts\\eval_d1.py --root C:\\nirikshak_data\\smart_invigilator
    python scripts\\eval_d1.py --root ... --max-images 200      # quick run
Writes outputs\\d1_eval\\report.md. Pose features are cached, so re-runs are fast.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import D1_CCTV_EXAM, OUTPUTS  # noqa: E402
from edge.ingest.sources import resolve_model  # noqa: E402
from edge.perception.pose_tracker import PoseTracker  # noqa: E402
from ml.eval.d1_posture import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(D1_CCTV_EXAM))
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-images", type=int, default=0, help="per split, 0 = all")
    ap.add_argument("--out", default=str(OUTPUTS / "d1_eval"))
    args = ap.parse_args()

    tracker = PoseTracker(resolve_model(args.model), imgsz=args.imgsz, conf=0.2)
    print(run(args.root, tracker, Path(args.out), args.max_images))
    print(f"\nreport -> {Path(args.out) / 'report.md'}")


if __name__ == "__main__":
    main()
