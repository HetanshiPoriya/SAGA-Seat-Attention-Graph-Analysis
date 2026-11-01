"""P-1: inspect D1 (CCTV Exam Monitor) — class names, counts per split, 5 sample images with boxes.

    python scripts\\inspect_d1.py [--root PATH] [--n 5] [--seed 0]
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import D1_CCTV_EXAM, OUTPUTS  # noqa: E402
from ml.datasets.cctv_exam import CCTVExamDataset  # noqa: E402

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
           (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61)]


def draw(ds, sample):
    img = cv2.imread(str(sample.image))
    H, W = img.shape[:2]
    for b in sample.boxes:
        color = PALETTE[b.cls % len(PALETTE)]
        p1, p2 = (int(b.x1 * W), int(b.y1 * H)), (int(b.x2 * W), int(b.y2 * H))
        cv2.rectangle(img, p1, p2, color, 2)
        text = ds.class_name(b.cls)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(img, text, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if sample.label is not None:
        banner = ds.class_name(sample.label)
    elif not sample.boxes:
        banner = "NO LABELS"
    else:
        banner = None
    if banner:
        cv2.putText(img, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(D1_CCTV_EXAM))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = CCTVExamDataset(args.root)
    print(f"Dataset: {ds.root}")
    print(f"Classes ({len(ds.names)}): {ds.names or '(none found)'}\n")

    print(f"{'split':<7}{'format':<8}{'images':>8}{'labelled':>10}{'boxes':>8}{'sources':>9}")
    for s in ds.splits.values():
        labelled = sum(1 for x in s.samples if x.boxes or x.label is not None)
        boxes = sum(len(x.boxes) for x in s.samples)
        # Roboflow names files "<source>_jpg.rf.<hash>.jpg"; augmented copies share <source>
        sources = len({x.image.name.split("_jpg.rf.")[0] for x in s.samples})
        print(f"{s.name:<7}{s.fmt:<8}{len(s.samples):>8}{labelled:>10}{boxes:>8}{sources:>9}")

    if ds.has_labels:
        print("\nPer-class counts (boxes, or images for folder layout):")
        for s in ds.splits.values():
            counts = s.class_counts()
            print(f"  {s.name}: " + ", ".join(f"{ds.class_name(c)}={n}" for c, n in sorted(counts.items())))
    else:
        print("\nWARNING: no annotations found (no data.yaml, labels/*.txt, _annotations.coco.json,\n"
              "_annotations.csv or class sub-folders). The Kaggle zip ships images only; download the\n"
              "labelled Roboflow version (YOLOv11 format) and pass it with --root. Samples are saved unlabelled.")

    out_dir = OUTPUTS / "d1_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = [(s.name, x) for s in ds.splits.values() for x in s.samples if x.boxes or x.label is not None]
    pool = pool or [(s.name, x) for s in ds.splits.values() for x in s.samples]
    rng = random.Random(args.seed)
    for i, (split, sample) in enumerate(rng.sample(pool, min(args.n, len(pool)))):
        out = out_dir / f"sample_{i}_{split}.jpg"
        cv2.imwrite(str(out), draw(ds, sample))
        print(f"saved {out}  ({Counter(ds.class_name(b.cls) for b in sample.boxes) or 'no boxes'})")


if __name__ == "__main__":
    main()
