"""P5 ablation: full-frame object detection vs hand/lap-ROI detection (spec §17.2 #6, §17.1 table).

The claim behind §6.6 is that cropping ROIs around the wrists and lap finds small objects that a
nano detector misses in the full frame. This script measures that on a labelled set: the same
weights, the same confidence threshold, once on whole images and once on the pose-derived ROIs.

    python scripts\\eval_objects.py --data C:\\SAGA_data\\exam_objects --model models\\objects.pt
    python scripts\\eval_objects.py --data C:\\SAGA_data\\exam_objects          # COCO weights: phone/book only

Reports, per class and overall: recall, precision and F1 at IoU 0.5, plus the median object size in
pixels (to show which objects the full frame loses). Images where no person is detected are counted
separately — the ROI path cannot see anything there, and that is part of its cost.
Writes outputs\\objects_eval\\report.md.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS  # noqa: E402
from edge.ingest.sources import resolve_model  # noqa: E402
from edge.perception.objects import ObjectDetector, iou  # noqa: E402
from edge.perception.pose_tracker import PoseTracker  # noqa: E402
from ml.datasets.objects import prepare  # noqa: E402


def load_gt(label_path: Path, names: list[str], w: int, h: int) -> list[tuple[str, np.ndarray]]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls, cx, cy, bw, bh = int(float(p[0])), *(float(x) for x in p[1:5])
        out.append((names[cls], np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                          (cx + bw / 2) * w, (cy + bh / 2) * h])))
    return out


def match(gt, dets, iou_thr=0.5):
    """Greedy one-to-one matching by confidence. Returns (tp, fp, fn) counts per class."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    used = set()
    for d in sorted(dets, key=lambda d: -d.conf):
        best, best_iou = None, iou_thr
        for i, (c, b) in enumerate(gt):
            if i in used or c != d.cls:
                continue
            o = iou(d.box, b)
            if o >= best_iou:
                best, best_iou = i, o
        if best is None:
            fp[d.cls] += 1
        else:
            used.add(best)
            tp[d.cls] += 1
    for i, (c, _) in enumerate(gt):
        if i not in used:
            fn[c] += 1
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return p, r, f


def count_only(args):
    """No object ground truth: report what each mode detects and what it costs. Counts are not
    recall — but if one mode finds objects the other never sees, that is still informative."""
    import random
    import time
    root = Path(args.data)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        sys.exit(f"no images under {root}")
    random.Random(0).shuffle(images)
    images = images[:args.limit]
    pose = PoseTracker(resolve_model(args.pose_model), imgsz=args.pose_imgsz)
    roi = ObjectDetector(resolve_model(args.model), imgsz=args.imgsz, conf=args.conf, wrist_k=args.roi_k)
    full = ObjectDetector(resolve_model(args.model), imgsz=args.full_imgsz or args.imgsz, conf=args.conf)
    counts = {"hand ROI": {}, "full frame": {}}
    secs = {"hand ROI": 0.0, "full frame": 0.0, "pose": 0.0}
    with_obj = {"hand ROI": 0, "full frame": 0}
    n_rois = 0
    for i, p in enumerate(images):
        img = cv2.imread(str(p))
        if img is None:
            continue
        t0 = time.perf_counter()
        people = pose.detect(img)
        secs["pose"] += time.perf_counter() - t0
        for mode, fn in (("hand ROI", lambda: roi(img, people)), ("full frame", lambda: full.full_frame(img))):
            t0 = time.perf_counter()
            dets = fn()
            secs[mode] += time.perf_counter() - t0
            for d in dets:
                counts[mode][d.cls] = counts[mode].get(d.cls, 0) + 1
            with_obj[mode] += any(d.prohibited for d in dets)
        n_rois += len(roi.rois)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(images)}")
    n = len(images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Object detection — hand/lap ROI vs full frame (detection counts, no object labels)\n",
          f"`{args.model}` · {n} images from `{root}` · conf ≥ {args.conf} · "
          f"ROI imgsz {args.imgsz}, full-frame imgsz {args.full_imgsz or args.imgsz}\n",
          f"Pose costs {secs['pose'] / n * 1000:.0f} ms/image; the ROI mode used "
          f"{n_rois / n:.1f} crops/image.\n",
          "| mode | ms/image | images with a prohibited object | detections by class |",
          "|---|---|---|---|"]
    for mode in counts:
        md.append(f"| {mode} | {secs[mode] / n * 1000:.0f} | {with_obj[mode]}/{n} | "
                  f"{', '.join(f'{k} {v}' for k, v in sorted(counts[mode].items())) or '–'} |")
    md.append("\nCounts, not recall: this set has no object boxes. Precision is unknown here too.\n")
    (out / "roi_vs_full.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nreport -> {out / 'roi_vs_full.md'}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of a YOLO export (data.yaml), or of images with --count-only")
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--pose-model", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--pose-imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--conf-sweep", type=float, nargs="*", metavar="T",
                    help="also report precision/recall at each of these thresholds")
    ap.add_argument("--limit", type=int, default=300, help="images to evaluate (CPU: keep it small)")
    ap.add_argument("--out", default=str(OUTPUTS / "objects_eval"))
    ap.add_argument("--count-only", action="store_true",
                    help="no object labels available: just count what each mode finds, and time it. "
                         "--data is then a folder of images (searched recursively).")
    ap.add_argument("--roi-k", type=float, default=1.4,
                    help="hand ROI width in shoulder widths (match the detector's training scale)")
    ap.add_argument("--full-imgsz", type=int, default=0,
                    help="imgsz for the full-frame baseline (default: same as --imgsz; use ~960 for equal cost)")
    args = ap.parse_args()

    if args.count_only:
        return count_only(args)
    data_yaml = prepare(args.data)
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = [cfg["names"][i] for i in sorted(cfg["names"])]
    img_dir = Path(cfg[args.split])
    lab_dir = img_dir.parent / "labels"      # <split>/images and <split>/labels sit side by side
    if not lab_dir.is_dir():
        sys.exit(f"no labels next to {img_dir} (expected {lab_dir})")
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.limit]
    if not images:
        sys.exit(f"no images in {img_dir}")
    if not any((lab_dir / f"{p.stem}.txt").exists() for p in images):
        sys.exit(f"no label file matches the images in {img_dir} — check {lab_dir}")

    det = ObjectDetector(resolve_model(args.model), imgsz=args.imgsz, conf=args.conf, wrist_k=args.roi_k)
    pose = PoseTracker(resolve_model(args.pose_model), imgsz=args.pose_imgsz)
    print(f"{len(images)} images · classes {names} · detector classes {sorted(set(det.map.values()))}")

    # Detect once at the lowest threshold, then score at each. Confidence calibration differs wildly
    # between a fine-tune and COCO weights, so the operating threshold has to be chosen, not assumed.
    thresholds = sorted(args.conf_sweep) if args.conf_sweep else [args.conf]
    det.conf = min(thresholds)
    sizes, no_person, per_image = [], 0, []
    for i, path in enumerate(images):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(lab_dir / f"{path.stem}.txt", names, w, h)
        gt = [(c, b) for c, b in gt if c in set(det.map.values())]   # classes this detector can predict
        sizes += [float(min(b[2] - b[0], b[3] - b[1])) for _, b in gt]
        people = pose.detect(img)
        no_person += not people
        per_image.append((gt, det.full_frame(img), det(img, people)))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(images)}")

    def score(thr: float):
        st = {m: (defaultdict(int), defaultdict(int), defaultdict(int)) for m in ("full frame", "hand ROI")}
        for gt, full, roi in per_image:
            for mode, dets in (("full frame", full), ("hand ROI", roi)):
                t, f, n = match(gt, [d for d in dets if d.conf >= thr])
                for acc, cur in zip(st[mode], (t, f, n)):
                    for k, v in cur.items():
                        acc[k] += v
        return st

    stats = score(thresholds[-1] if len(thresholds) == 1 else max(thresholds))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Object detection — full frame vs hand/lap ROI\n",
          f"`{args.model}` · {len(images)} images from `{args.data}` ({args.split}) · conf ≥ {args.conf} · "
          f"IoU 0.5 · object imgsz {args.imgsz}\n",
          f"Median labelled object size: {np.median(sizes):.0f} px (min side); "
          f"{no_person}/{len(images)} images had no detected person (ROI mode is blind there).\n",
          "| mode | class | TP | FP | FN | precision | recall | F1 |", "|---|---|---|---|---|---|---|---|"]
    for mode, (tp, fp, fn) in stats.items():
        classes = sorted(set(tp) | set(fp) | set(fn))
        for c in classes + ["ALL"]:
            if c == "ALL":
                t, f_, n = sum(tp.values()), sum(fp.values()), sum(fn.values())
            else:
                t, f_, n = tp[c], fp[c], fn[c]
            p, r, f1 = prf(t, f_, n)
            md.append(f"| {mode} | {c} | {t} | {f_} | {n} | {p:.3f} | {r:.3f} | {f1:.3f} |")
    if len(thresholds) > 1:
        md += ["\n## Operating threshold\n",
               "| conf | mode | class | precision | recall | F1 |", "|---|---|---|---|---|---|"]
        for thr in thresholds:
            st = score(thr)
            for mode, (tp, fp, fn) in st.items():
                for c in sorted(set(tp) | set(fp) | set(fn)) + ["ALL"]:
                    if c == "ALL":
                        t, f_, n = sum(tp.values()), sum(fp.values()), sum(fn.values())
                    else:
                        t, f_, n = tp[c], fp[c], fn[c]
                    p, r, f1 = prf(t, f_, n)
                    md.append(f"| {thr} | {mode} | {c} | {p:.3f} | {r:.3f} | {f1:.3f} |")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nreport -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
