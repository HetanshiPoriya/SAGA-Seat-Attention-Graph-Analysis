"""P5: fine-tune the exam-object detector (spec §18 `ml/train_objects.py`, data plan D5).

COCO weights already give `phone` (cell phone), `book` and `bottle`, which is enough to demo
phone-in-hand and phone-in-lap. They do NOT have `chit` (paper note), `earphone` or `smartwatch` —
for those the detector has to be trained on a labelled set (Roboflow Universe exam / phone / paper
sets; paper is the hardest class, see spec §2.2 on AutoOEP).

    # 1. download a YOLOv11 export into C:\\SAGA_data\\exam_objects  (images + labels + data.yaml)
    # 2. train (CPU is slow: ~1-2 h for 50 epochs at 320 px on a few thousand images)
    python scripts\\train_objects.py --data C:\\SAGA_data\\exam_objects --epochs 50 --imgsz 320
    # 3. the best weights are copied to models\\objects.pt, which run_seats picks up:
    python scripts\\run_seats.py --source 0 --auto-seats 30 --objects --obj-model models\\objects.pt

Training crops are small object close-ups, which matches how the detector is used at inference
(hand/lap ROIs, not whole frames) — keep --imgsz the same in both places.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS, OUTPUTS  # noqa: E402
from edge.ingest.sources import resolve_model  # noqa: E402
from ml.datasets.objects import prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", metavar="FOLDER",
                    help="folder(s) of YOLO exports (each contains data.yaml); several are merged")
    ap.add_argument("--model", default="yolo11n.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=320, help="match run_seats --obj-imgsz")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--classes", nargs="*", help="limit to these SAGA classes (default: all that map)")
    ap.add_argument("--out", default=str(MODELS / "objects.pt"))
    ap.add_argument("--name", default="objects")
    ap.add_argument("--prepared", help="where to write the remapped dataset (default: next to --data)")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    # A class subset produces different labels, so it needs its own folder — otherwise the next run
    # silently trains on the previous run's label files.
    out_dir = args.prepared
    if out_dir is None and args.classes:
        first = Path(args.data[0])
        stem = first.name if len(args.data) == 1 else "objects_merged"
        out_dir = first.parent / f"{stem}_saga_{'-'.join(args.classes)}"
    data_yaml = prepare(args.data, out=out_dir, classes=args.classes)
    if args.prepare_only:
        return
    from ultralytics import YOLO
    model = YOLO(resolve_model(args.model))
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=str(OUTPUTS / "train"), name=args.name, exist_ok=True, patience=15)
    m = model.val(data=str(data_yaml), imgsz=args.imgsz, split="val")
    print(f"\nval mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")
    for i, c in enumerate(m.names.values()):
        if i < len(m.box.ap50):
            print(f"  {c:<12} AP50 {m.box.ap50[i]:.3f}")
    best = Path(model.trainer.best)
    shutil.copy(best, args.out)
    print(f"\nbest weights -> {args.out}\n"
          f"Next: python scripts\\eval_objects.py --data {args.data[0]} --model {args.out}  "
          f"(full-frame vs hand-ROI ablation)")


if __name__ == "__main__":
    main()
