"""Loader for D1 — CCTV Exam Monitor Dataset (Roboflow export, via Kaggle).

The export format is auto-detected per split, since Roboflow can ship any of:

  yolo   <split>/labels/*.txt (+ <split>/images/), or .txt next to each image,
         class names from data.yaml at the dataset root
  coco   <split>/_annotations.coco.json
  csv    <split>/_annotations.csv   (filename,width,height,class,xmin,ymin,xmax,ymax)
  folder <split>/<ClassName>/*.jpg   (classification layout, no boxes)
  none   images only — no annotations found

Boxes are stored normalised (0-1, xyxy) so no image has to be opened to load labels.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


@dataclass
class Box:
    cls: int
    x1: float  # normalised 0-1
    y1: float
    x2: float
    y2: float


@dataclass
class Sample:
    image: Path
    boxes: list[Box] = field(default_factory=list)
    label: int | None = None  # image-level class (folder layout only)


@dataclass
class Split:
    name: str
    fmt: str
    samples: list[Sample]

    def class_counts(self) -> Counter:
        c = Counter()
        for s in self.samples:
            if s.label is not None:
                c[s.label] += 1
            c.update(b.cls for b in s.boxes)
        return c


class CCTVExamDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset not found: {self.root}")
        self.names: list[str] = self._names_from_yaml()
        self.splits: dict[str, Split] = {}
        for name in SPLITS:
            d = self.root / name
            if d.is_dir():
                self.splits[name] = self._load_split(d)

    # ---------- class names ----------
    def _names_from_yaml(self) -> list[str]:
        for p in (self.root / "data.yaml", self.root / "data.yml"):
            if p.exists():
                names = yaml.safe_load(p.read_text()).get("names", [])
                if isinstance(names, dict):  # {0: 'a', 1: 'b'}
                    names = [names[k] for k in sorted(names, key=int)]
                return list(names)
        return []

    def _class_id(self, name: str) -> int:
        if name not in self.names:
            self.names.append(name)
        return self.names.index(name)

    # ---------- split loading ----------
    def _load_split(self, d: Path) -> Split:
        if (d / "_annotations.coco.json").exists():
            return Split(d.name, "coco", self._load_coco(d))
        if (d / "_annotations.csv").exists():
            return Split(d.name, "csv", self._load_csv(d))

        img_dir = d / "images" if (d / "images").is_dir() else d
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        label_dir = d / "labels" if (d / "labels").is_dir() else img_dir
        if images and any((label_dir / f"{p.stem}.txt").exists() for p in images[:50]):
            return Split(d.name, "yolo", [self._load_yolo(p, label_dir) for p in images])

        subdirs = [s for s in d.iterdir() if s.is_dir() and s.name not in ("images", "labels")]
        if not images and subdirs:
            return Split(d.name, "folder", self._load_folders(subdirs))

        return Split(d.name, "none", [Sample(p) for p in images])

    def _load_yolo(self, img: Path, label_dir: Path) -> Sample:
        s = Sample(img)
        txt = label_dir / f"{img.stem}.txt"
        if not txt.exists():
            return s
        for line in txt.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            cls, nums = int(v[0]), list(map(float, v[1:]))
            if len(nums) == 4:  # cx cy w h
                cx, cy, w, h = nums
                s.boxes.append(Box(cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            else:  # polygon x1 y1 x2 y2 ... -> bounding box
                xs, ys = nums[0::2], nums[1::2]
                s.boxes.append(Box(cls, min(xs), min(ys), max(xs), max(ys)))
        return s

    def _load_coco(self, d: Path) -> list[Sample]:
        coco = json.loads((d / "_annotations.coco.json").read_text())
        cat_ids = {c["id"]: self._class_id(c["name"]) for c in coco["categories"]}
        imgs = {im["id"]: im for im in coco["images"]}
        samples = {iid: Sample(d / im["file_name"]) for iid, im in imgs.items()}
        for a in coco["annotations"]:
            im = imgs[a["image_id"]]
            x, y, w, h = a["bbox"]
            W, H = im["width"], im["height"]
            samples[a["image_id"]].boxes.append(
                Box(cat_ids[a["category_id"]], x / W, y / H, (x + w) / W, (y + h) / H))
        return list(samples.values())

    def _load_csv(self, d: Path) -> list[Sample]:
        samples: dict[str, Sample] = {}
        with open(d / "_annotations.csv", newline="") as f:
            for r in csv.DictReader(f):
                s = samples.setdefault(r["filename"], Sample(d / r["filename"]))
                W, H = float(r["width"]), float(r["height"])
                s.boxes.append(Box(self._class_id(r["class"]),
                                   float(r["xmin"]) / W, float(r["ymin"]) / H,
                                   float(r["xmax"]) / W, float(r["ymax"]) / H))
        return list(samples.values())

    def _load_folders(self, subdirs: list[Path]) -> list[Sample]:
        out = []
        for sd in sorted(subdirs):
            cid = self._class_id(sd.name)
            out += [Sample(p, label=cid) for p in sorted(sd.iterdir())
                    if p.suffix.lower() in IMAGE_EXTS]
        return out

    # ---------- helpers ----------
    def class_name(self, cid: int) -> str:
        return self.names[cid] if 0 <= cid < len(self.names) else f"class_{cid}"

    @property
    def has_labels(self) -> bool:
        return any(s.fmt != "none" for s in self.splits.values())
