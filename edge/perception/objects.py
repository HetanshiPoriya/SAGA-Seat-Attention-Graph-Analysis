"""Hand/desk-ROI object detection (spec §6.6, P5).

A phone in a wide CCTV frame is 15–30 px across: a nano detector run on the full frame misses it.
SAGA instead crops small ROIs from the pose keypoints — one around each visible wrist, one over the
lap/desk area in front of the student — and runs the detector on those crops, where the object fills
a useful share of the pixels. Boxes are mapped back to full-frame coordinates, so everything
downstream (seat assignment, interaction features, transfer) works in image space as usual.

Two sets of weights are supported:

* `yolo11n.pt` (COCO, no training needed) — gives `phone` (COCO *cell phone*) and `book`/`bottle` as
  allowed objects. Enough to demo phone-in-lap and phone-in-hand today.
* `models/objects.pt` — fine-tuned on a labelled exam-object set (Roboflow phone/paper/earphone,
  see scripts/train_objects.py). Needed for `chit` (paper) and `earphone`/`smartwatch`: COCO has no
  such classes, and paper is the class object detectors are worst at (AutoOEP, spec §2.2).

Cost on CPU: one nano forward pass per ROI. Callers run this every N frames (spec §6.1 wants objects
at 3–5 fps, pose at 10–15) and reuse the last result in between.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from edge.perception.head_pose import LHIP, LSH, LWRIST, RHIP, RSH, RWRIST

# What SAGA reasons about. `chit` = paper note; allowed objects are detected on purpose, so that
# "there is an object near the hand" does not fire on a water bottle or the question paper.
PROHIBITED = ("phone", "chit", "earphone", "smartwatch", "calculator")
ALLOWED = ("book", "bottle", "pen")
CLASSES = PROHIBITED + ALLOWED

# COCO names -> SAGA classes (for untrained yolo11n.pt). COCO has no chit/earphone/watch.
COCO_MAP = {"cell phone": "phone", "book": "book", "bottle": "bottle"}

OBJ_COLORS = {True: (0, 0, 255), False: (120, 200, 120)}  # prohibited / allowed


@dataclass
class ObjectDet:
    cls: str
    conf: float
    box: np.ndarray                 # xyxy, full-frame pixels
    track_id: int | None = None
    roi_kind: str = "frame"         # wrist | lap | frame (which ROI it was found in)

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def prohibited(self) -> bool:
        return self.cls in PROHIBITED


# ------------------------------------------------------------------ ROIs from keypoints
def _box(cx, cy, half_w, half_h, shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([max(cx - half_w, 0), max(cy - half_h, 0), min(cx + half_w, w - 1), min(cy + half_h, h - 1)],
                    dtype=float)


def _scale(person, thr: float) -> float:
    """Shoulder width in pixels — the natural unit for everything on a body."""
    v = person.kpt_conf >= thr
    if v[LSH] and v[RSH]:
        return float(np.hypot(*(person.kpts[RSH] - person.kpts[LSH])))
    return float(person.box[2] - person.box[0]) * 0.5  # fallback: half the detection box width


def wrist_rois(person, shape, thr=0.4, k=1.4) -> list[np.ndarray]:
    """A square ROI per visible wrist, k shoulder-widths across, pushed slightly down: a held object
    sits in front of / below the wrist joint."""
    sw = _scale(person, thr)
    if sw < 4:
        return []
    half = k * sw / 2
    out = []
    for w in (LWRIST, RWRIST):
        if person.kpt_conf[w] >= thr:
            x, y = person.kpts[w]
            out.append(_box(x, y + 0.15 * sw, half, half, shape))
    return out


def lap_roi(person, shape, thr=0.4) -> np.ndarray | None:
    """The lap / desk area in front of the student: where a phone held low or a chit on the knee is.
    Anchored on the hips when visible, otherwise projected down from the shoulders — so it still
    exists when the desk hides the lower body (then it covers the desk surface instead)."""
    v = person.kpt_conf >= thr
    sw = _scale(person, thr)
    if sw < 4 or not (v[LSH] and v[RSH]):
        return None
    sh_x = float((person.kpts[LSH, 0] + person.kpts[RSH, 0]) / 2)
    sh_y = float((person.kpts[LSH, 1] + person.kpts[RSH, 1]) / 2)
    if v[LHIP] and v[RHIP]:
        cy = float((person.kpts[LHIP, 1] + person.kpts[RHIP, 1]) / 2) + 0.35 * sw
    else:
        cy = sh_y + 1.5 * sw
    return _box(sh_x, cy, 0.9 * sw, 0.8 * sw, shape)


def person_rois(person, shape, thr=0.4, wrist_k=1.4) -> list[tuple[str, np.ndarray]]:
    rois = [("wrist", b) for b in wrist_rois(person, shape, thr, wrist_k)]
    lap = lap_roi(person, shape, thr)
    if lap is not None:
        rois.append(("lap", lap))
    return _merge(rois)


def _merge(rois: list[tuple[str, np.ndarray]], iou_thr=0.55) -> list[tuple[str, np.ndarray]]:
    """Merge heavily overlapping ROIs (two hands together) into one crop — one inference instead of
    two. The merged ROI keeps the first kind, which is only used for reporting."""
    out: list[tuple[str, np.ndarray]] = []
    for kind, b in rois:
        for i, (k0, b0) in enumerate(out):
            if iou(b, b0) > iou_thr:
                out[i] = (k0, np.array([min(b0[0], b[0]), min(b0[1], b[1]),
                                        max(b0[2], b[2]), max(b0[3], b[3])]))
                break
        else:
            out.append((kind, b))
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / area) if area > 0 else 0.0


# ------------------------------------------------------------------ detector
class ObjectDetector:
    def __init__(self, model: str = "yolo11n.pt", imgsz: int = 320, conf: float = 0.30,
                 device: str | None = None, max_rois: int = 16, wrist_k: float = 1.4,
                 class_map: dict[str, str] | None = None, min_roi_px: int = 24):
        self.model = YOLO(model)
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        self.imgsz, self.conf, self.max_rois, self.wrist_k = imgsz, conf, max_rois, wrist_k
        self.min_roi_px = min_roi_px
        names = set(getattr(self.model, "names", {}).values())
        # fine-tuned weights already use SAGA class names; COCO weights need the alias map. COCO has
        # both kinds of name ("book"/"bottle" straight, "cell phone" -> phone), so merge the two.
        self.map = class_map if class_map is not None else {
            **{n: n for n in names if n in CLASSES},
            **{k: v for k, v in COCO_MAP.items() if k in names}}
        self.classes = sorted({i for i, n in getattr(self.model, "names", {}).items() if n in self.map})
        self.rois: list[tuple[str, np.ndarray]] = []   # last frame's ROIs, for drawing

    # ---------------------------------------------------------- inference
    def __call__(self, frame: np.ndarray, people: list) -> list[ObjectDet]:
        """Detect objects inside the hand/lap ROIs of `people`. Returns full-frame boxes."""
        rois: list[tuple[str, np.ndarray]] = []
        for p in people:
            rois += person_rois(p, frame.shape, wrist_k=self.wrist_k)
        rois = [(k, b) for k, b in _merge(rois)
                if b[2] - b[0] >= self.min_roi_px and b[3] - b[1] >= self.min_roi_px][:self.max_rois]
        self.rois = rois
        if not rois:
            return []
        crops = [frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for _, b in rois]
        dets = []
        for (kind, b), res in zip(rois, self._predict(crops)):
            for d in res:
                d.box[[0, 2]] += b[0]
                d.box[[1, 3]] += b[1]
                d.roi_kind = kind
                dets.append(d)
        return nms(dets)

    def full_frame(self, frame: np.ndarray) -> list[ObjectDet]:
        """Baseline for the hand-ROI ablation (spec §17.2 #6): the whole frame in one pass."""
        self.rois = [("frame", np.array([0, 0, frame.shape[1] - 1, frame.shape[0] - 1], float))]
        return nms(self._predict([frame])[0])

    def _predict(self, images: list[np.ndarray]) -> list[list[ObjectDet]]:
        kw = dict(imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        if self.classes:
            kw["classes"] = self.classes
        out = []
        for r in self.model.predict(images, **kw):
            dets = []
            if r.boxes is not None and len(r.boxes):
                names = r.names
                for box, c, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.int().cpu().tolist()):
                    saga = self.map.get(names[cls])
                    if saga:
                        dets.append(ObjectDet(saga, float(c), box.astype(float)))
            out.append(dets)
        return out


def nms(dets: list[ObjectDet], iou_thr: float = 0.5) -> list[ObjectDet]:
    """Same object seen in two overlapping ROIs -> keep the most confident box."""
    keep: list[ObjectDet] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if not any(k.cls == d.cls and iou(k.box, d.box) > iou_thr for k in keep):
            keep.append(d)
    return keep


# ------------------------------------------------------------------ tracking
@dataclass
class _Track:
    id: int
    cls: str
    box: np.ndarray
    last_t: float
    hits: int = 1
    seats: list[str] = field(default_factory=list)


class ObjectTracker:
    """Minimal IoU + centre-distance tracker. Object tracks are what makes a *transfer* detectable:
    one id that is held by seat i and then by seat j (reasoner/transfer.py). Deliberately simple —
    objects are slow and few, and a wrong merge is safer than a new id per frame."""

    def __init__(self, max_age_s: float = 1.5, iou_thr: float = 0.2, max_move: float = 2.0):
        self.max_age_s, self.iou_thr, self.max_move = max_age_s, iou_thr, max_move
        self.tracks: dict[int, _Track] = {}
        self._next = 1

    def update(self, t: float, dets: list[ObjectDet], scale: float = 60.0) -> list[ObjectDet]:
        """Assign track ids in place. `scale` = a typical shoulder width in px (movement budget)."""
        for tid in [i for i, tr in self.tracks.items() if t - tr.last_t > self.max_age_s]:
            del self.tracks[tid]
        free = dict(self.tracks)
        for d in sorted(dets, key=lambda d: -d.conf):
            best, best_key = None, None
            for tid, tr in free.items():
                if tr.cls != d.cls:
                    continue
                move = float(np.hypot(*(d.center - np.array([(tr.box[0] + tr.box[2]) / 2,
                                                             (tr.box[1] + tr.box[3]) / 2]))))
                ov = iou(d.box, tr.box)
                if ov < self.iou_thr and move > self.max_move * max(scale, 1.0):
                    continue
                key = (-ov, move)
                if best_key is None or key < best_key:
                    best, best_key = tid, key
            if best is None:
                best = self._next
                self._next += 1
                self.tracks[best] = _Track(best, d.cls, d.box.copy(), t)
            else:
                tr = self.tracks[best]
                tr.box, tr.last_t, tr.hits = d.box.copy(), t, tr.hits + 1
                free.pop(best, None)
            d.track_id = best
        return dets


# ------------------------------------------------------------------ drawing
def draw_objects(frame, dets: list[ObjectDet], rois: list[tuple[str, np.ndarray]] | None = None):
    from edge.viz import label, ui_scale
    s = ui_scale(frame)
    for _, b in rois or []:
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (70, 70, 70), max(1, int(s)))
    for d in dets:
        color = OBJ_COLORS[d.prohibited]
        x1, y1, x2, y2 = d.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, int(2 * s)))
        tag = f"{d.cls}" + (f" #{d.track_id}" if d.track_id else "") + f" {d.conf:.2f}"
        label(frame, tag, (x1, y1), color, 0.45)
    return frame
