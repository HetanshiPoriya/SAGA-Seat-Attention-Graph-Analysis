"""P1 evaluation: do head-pose + torso features separate the D1 posture classes? (spec §17.1)

For every labelled box: run YOLO pose on the image, match the box to a detected person (IoU),
compute the T1 head-pose/body features, then
  (a) rule baseline — interpretable thresholds (yaw sign/size, facing away, standing geometry),
      tuned by grid search on train;
  (b) learned — gradient-boosted trees on the same features, trained on train.
Both are scored on valid and test: accuracy, macro-F1, per-class report, confusion matrix.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import re
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from edge.perception.head_pose import pose_features
from ml.datasets.cctv_exam import CCTVExamDataset

FEATURES = ["yaw", "pitch", "facing_camera", "facing_away", "facing_profile", "head_px", "kp_conf_upper",
            "shoulder_w", "shoulder_tilt", "lean_x", "torso_len", "shoulder_ratio", "knee_drop", "box_aspect"]
CANON = [("correct", "correct"), ("left", "left"), ("right", "right"), ("forward", "forward"),
         ("backward", "backward"), ("back", "backward"), ("stand", "stand")]


def canonical(name: str) -> str:
    n = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    return next((c for key, c in CANON if key in n), n)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- feature extraction
def extract(ds: CCTVExamDataset, split: str, tracker, cache_dir: Path, min_iou=0.3, max_images=0) -> list[dict]:
    key = hashlib.md5(f"{ds.root}|{tracker.model.ckpt_path}|{tracker.imgsz}|{max_images}".encode()).hexdigest()[:8]
    cache = cache_dir / f"features_{split}_{key}.csv"
    if cache.exists():
        with open(cache, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in FEATURES + ["matched"]:
                r[k] = float(r[k]) if r[k] != "" else float("nan")
        return rows

    samples = [s for s in ds.splits[split].samples if s.boxes]
    if max_images:
        samples = samples[:max_images]
    rows = []
    for i, s in enumerate(samples):
        img = cv2.imread(str(s.image))
        H, W = img.shape[:2]
        people = tracker.detect(img)
        gts = [(b, [b.x1 * W, b.y1 * H, b.x2 * W, b.y2 * H]) for b in s.boxes]
        pairs = sorted(((iou(g, p.box), gi, pi) for gi, (_, g) in enumerate(gts) for pi, p in enumerate(people)),
                       reverse=True)
        used_g, used_p, match = set(), set(), {}
        for v, gi, pi in pairs:
            if v >= min_iou and gi not in used_g and pi not in used_p:
                used_g.add(gi), used_p.add(pi)
                match[gi] = pi
        for gi, (b, _) in enumerate(gts):
            row = {"image": s.image.name, "cls": ds.class_name(b.cls), "matched": float(gi in match)}
            f = (pose_features(people[match[gi]].kpts, people[match[gi]].kpt_conf, people[match[gi]].box)
                 if gi in match else {})
            row.update({k: f.get(k, float("nan")) for k in FEATURES})
            rows.append(row)
        if (i + 1) % 100 == 0 or i + 1 == len(samples):
            print(f"\r  {split}: {i + 1}/{len(samples)} images", end="", flush=True)
    print()

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", newline="") as f:
        w = csv.DictWriter(f, ["image", "cls", "matched"] + FEATURES)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()})
    return rows


def to_xy(rows):
    rows = [r for r in rows if r["matched"] == 1.0]
    X = np.array([[r[k] for k in FEATURES] for r in rows], float)
    y = np.array([canonical(r["cls"]) for r in rows])
    return X, y


# ---------------------------------------------------------------- (a) rule baseline
def rule_predict(X, classes, t_yaw, sign, p_down, a_stand):
    f = {k: X[:, i] for i, k in enumerate(FEATURES)}
    yaw = np.nan_to_num(f["yaw"])
    pitch = np.nan_to_num(f["pitch"])
    default = "correct" if "correct" in classes else classes[0]
    pred = np.full(len(X), default, dtype=object)
    if "forward" in classes:
        pred[pitch < -p_down] = "forward"
    if "left" in classes and "right" in classes:
        pred[sign * yaw > t_yaw] = "right"
        pred[sign * yaw < -t_yaw] = "left"
    if "backward" in classes:
        pred[f["facing_away"] == 1] = "backward"
    if "stand" in classes:
        standing = (np.nan_to_num(f["knee_drop"]) > 0.6) & (f["box_aspect"] > a_stand)
        pred[standing] = "stand"
    return pred


def tune_rules(X, y, classes):
    grid = itertools.product([10, 15, 20, 25, 30, 40], [1, -1], [15, 25, 35, 1e9], [1.5, 2.0, 2.5, 3.0, 1e9])
    best = max(grid, key=lambda g: f1_score(y, rule_predict(X, classes, *g), average="macro", zero_division=0))
    return dict(zip(["t_yaw", "sign", "p_down", "a_stand"], best))


# ---------------------------------------------------------------- report
def score(name, y, pred, labels) -> str:
    cm = confusion_matrix(y, pred, labels=labels)
    head = "true \\ pred".ljust(12) + "".join(l[:9].rjust(10) for l in labels)
    lines = [head] + [l[:11].ljust(12) + "".join(str(v).rjust(10) for v in row) for l, row in zip(labels, cm)]
    return (f"### {name}\naccuracy {accuracy_score(y, pred):.3f} · macro-F1 "
            f"{f1_score(y, pred, labels=labels, average='macro', zero_division=0):.3f}\n\n```\n"
            + classification_report(y, pred, labels=labels, zero_division=0, digits=3)
            + "\n" + "\n".join(lines) + "\n```\n")


def frame_group(image_name: str, block: int = 200) -> str:
    """Images cut from video are named '<prefix><frame>_jpg.rf.<hash>'. Neighbouring frames are near
    duplicates, so a random split leaks; group them into blocks of `block` frames per video prefix."""
    m = re.match(r"([A-Za-z]*)(\d+)", image_name)
    return f"{m.group(1)}{int(m.group(2)) // block}" if m else image_name.split("_jpg.rf.")[0]


def grouped_cv(data, folds=5, block=200) -> str:
    rows = [r for split_rows in data.values() for r in split_rows if r["matched"] == 1.0]
    X, y = to_xy(rows)
    groups = np.array([frame_group(r["image"], block) for r in rows])
    out = [f"\n## Leakage-safe estimate ({folds}-fold CV, all splits pooled, frames grouped in blocks of {block})\n",
           "Roboflow's random split puts neighbouring video frames in both train and test, which inflates "
           "the learned model's split scores above. Report these numbers instead.\n",
           "| classes | rules acc | rules macro-F1 | learned acc | learned macro-F1 |", "|---|---|---|---|---|"]
    for name, keep in [("all", np.ones(len(y), bool)), ("head/posture only", y != "passingnotes")]:
        if name != "all" and keep.all():
            continue
        Xk, yk, gk = X[keep], y[keep], groups[keep]
        if len(set(gk)) < folds:
            out.append(f"| {name} | too few frame groups for CV | | | |")
            continue
        s = np.zeros(4)
        for tr, te in GroupKFold(folds).split(Xk, yk, gk):
            labels = sorted(str(c) for c in set(yk[tr]))
            rp = rule_predict(Xk[te], labels, **tune_rules(Xk[tr], yk[tr], labels))
            lp = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced",
                                                random_state=0).fit(Xk[tr], yk[tr]).predict(Xk[te])
            s += [accuracy_score(yk[te], rp), f1_score(yk[te], rp, average="macro", zero_division=0),
                  accuracy_score(yk[te], lp), f1_score(yk[te], lp, average="macro", zero_division=0)]
        s /= folds
        out.append(f"| {name} | {s[0]:.3f} | {s[1]:.3f} | {s[2]:.3f} | {s[3]:.3f} |")
    return "\n".join(out) + "\n"


def run(root, tracker, out_dir: Path, max_images=0) -> str:
    ds = CCTVExamDataset(root)
    if not ds.has_labels:
        raise SystemExit(f"No labels found in {root} — see scripts\\inspect_d1.py for what is expected.")
    data = {}
    for split in ds.splits:
        rows = extract(ds, split, tracker, out_dir, max_images=max_images)
        data[split] = rows

    report = [f"# D1 posture evaluation\n\ndataset `{ds.root}` · model `{Path(tracker.model.ckpt_path).name}` "
              f"· imgsz {tracker.imgsz}\n\n## Detection match rate (GT box ↔ pose person, IoU ≥ 0.3)\n"]
    for split, rows in data.items():
        m = sum(r["matched"] == 1.0 for r in rows)
        report.append(f"- {split}: {m}/{len(rows)} boxes ({m / max(len(rows), 1):.1%})")

    train = "train" if "train" in data else next(iter(data))
    Xtr, ytr = to_xy(data[train])
    labels = sorted(str(c) for c in set(ytr))
    params = tune_rules(Xtr, ytr, labels)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced",
                                         random_state=0).fit(Xtr, ytr)
    report.append(f"\nClasses (canonical): {labels}\n\nRule thresholds tuned on {train}: {params}\n"
                  f"(sign=+1 means positive image-yaw is labelled 'right')\n")
    for split in [s for s in ("valid", "test") if s in data] or [train]:
        X, y = to_xy(data[split])
        report.append(f"\n## {split} ({len(y)} matched boxes)\n")
        report.append(score("Rule baseline", y, rule_predict(X, labels, **params), labels))
        report.append(score("Learned (gradient-boosted trees)", y, clf.predict(X), labels))
    report.append(grouped_cv(data))
    report.append("\nNote: Roboflow train augmentations may include horizontal flips, which would make "
                  "left/right labels noisy in train. Valid/test are not augmented.\n")
    text = "\n".join(report)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return text
