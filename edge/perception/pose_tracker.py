"""YOLO pose + BoT-SORT tracking (spec §6.3). One forward pass -> boxes, 17 COCO keypoints, track IDs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

BOTSORT_CFG = Path(__file__).with_name("botsort.yaml")

# COCO-17 skeleton: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows,
# 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
LIMB_COLORS = {"head": (255, 200, 0), "arm": (0, 200, 255), "torso": (0, 255, 120), "leg": (200, 120, 255)}


def _limb_kind(a: int, b: int) -> str:
    if a <= 4 and b <= 4:
        return "head"
    if {a, b} & {7, 8, 9, 10}:
        return "arm"
    if {a, b} & {13, 14, 15, 16}:
        return "leg"
    return "torso"


@dataclass
class Person:
    track_id: int | None   # None until the tracker confirms the track
    box: np.ndarray        # (4,) xyxy pixels
    conf: float
    kpts: np.ndarray       # (17, 2) pixels
    kpt_conf: np.ndarray   # (17,)


class PoseTracker:
    def __init__(self, model: str = "yolo11n-pose.pt", device: str | None = None,
                 imgsz: int = 640, conf: float = 0.25, tracker: str | Path = BOTSORT_CFG):
        self.model = YOLO(model)
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        self.imgsz, self.conf, self.tracker = imgsz, conf, str(tracker)

    def __call__(self, frame: np.ndarray) -> list[Person]:
        """Pose + tracking (video). Track IDs persist across calls."""
        r = self.model.track(frame, persist=True, tracker=self.tracker, imgsz=self.imgsz,
                             conf=self.conf, device=self.device, classes=[0], verbose=False)[0]
        return self._parse(r)

    def detect(self, image: np.ndarray) -> list[Person]:
        """Pose only, no tracking (still images). track_id is None."""
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, device=self.device,
                               classes=[0], verbose=False)[0]
        return self._parse(r)

    @staticmethod
    def _parse(r) -> list[Person]:
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(boxes)
        kxy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
        return [Person(ids[i], boxes[i], float(confs[i]), kxy[i], kc[i]) for i in range(len(boxes))]

    def reset(self):
        """Clear tracker state (call when switching to a new video)."""
        for t in getattr(self.model.predictor, "trackers", None) or []:
            t.reset()


def id_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (160, 160, 160)
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def draw_people(frame: np.ndarray, people: list[Person], kpt_thresh: float = 0.4,
                draw_boxes: bool = True) -> np.ndarray:
    for p in people:
        color = id_color(p.track_id)
        ok = p.kpt_conf >= kpt_thresh
        pts = p.kpts.astype(int)
        for a, b in SKELETON:
            if ok[a] and ok[b]:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), LIMB_COLORS[_limb_kind(a, b)], 2, cv2.LINE_AA)
        for i in np.flatnonzero(ok):
            cv2.circle(frame, tuple(pts[i]), 3, color, -1, cv2.LINE_AA)

        x1, y1, x2, y2 = p.box.astype(int)
        if draw_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"#{p.track_id}" if p.track_id is not None else "?"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame
