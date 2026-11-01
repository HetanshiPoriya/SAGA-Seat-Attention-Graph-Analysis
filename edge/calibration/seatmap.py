"""Seat map: seat polygons in the image + homography to a top-down seat plan (spec §6.2).

Two ways to build one:
  grid  — invigilator clicks the 4 corners of the desk area, gives rows × cols. Corner order is
          front-left, front-right, back-right, back-left as seen in the image; row A is the front.
          H maps image -> plan, where seat (r, c) is the unit square [c, c+1] × [r, r+1].
  auto  — "pseudo seat map" (spec §16.3 step 2): one seat per stable person found in the first
          frames; rows are grouped by image height. No physical calibration needed (for D2/SAV clips).

Seats are labelled A1, A2, … (row letter from the front, column number from image-left).
"""
from __future__ import annotations

import json
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from edge.perception.head_pose import LHIP, LSH, RHIP, RSH, body_features


def row_label(r: int) -> str:
    return string.ascii_uppercase[r] if r < 26 else f"R{r + 1}"


@dataclass
class Seat:
    label: str
    row: int
    col: int
    polygon: list[list[float]]   # image px
    center: list[float]          # image px
    neighbours: list[str] = field(default_factory=list)


@dataclass
class SeatMap:
    mode: str                                 # grid | auto
    image_size: list[int]                     # [w, h]
    seats: list[Seat]
    rows: int
    cols: int
    corners: list[list[float]] | None = None  # grid only, image px
    H: list[list[float]] | None = None        # grid only, image -> plan

    def __post_init__(self):
        self._by_label = {s.label: s for s in self.seats}
        self._polys = [np.asarray(s.polygon, np.float32) for s in self.seats]

    # ---------- construction ----------
    @classmethod
    def from_grid(cls, corners, rows: int, cols: int, image_size) -> "SeatMap":
        src = np.asarray(corners, np.float32)
        dst = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
        H = cv2.getPerspectiveTransform(src, dst)
        Hinv = np.linalg.inv(H)
        seats = []
        for r in range(rows):
            for c in range(cols):
                cell = np.float32([[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]])
                poly = cv2.perspectiveTransform(cell[None], Hinv)[0]
                centre = cv2.perspectiveTransform(np.float32([[[c + 0.5, r + 0.5]]]), Hinv)[0, 0]
                nbrs = [f"{row_label(rr)}{cc + 1}" for rr in (r - 1, r, r + 1) for cc in (c - 1, c, c + 1)
                        if (rr, cc) != (r, c) and 0 <= rr < rows and 0 <= cc < cols]
                seats.append(Seat(f"{row_label(r)}{c + 1}", r, c, poly.tolist(), centre.tolist(), nbrs))
        return cls("grid", list(image_size), seats, rows, cols,
                   src.tolist(), H.tolist())

    @classmethod
    def from_anchors(cls, anchors: list[tuple[float, float]], heights: list[float], image_size) -> "SeatMap":
        """One seat per anchor point; heights = person box height, used for row grouping + seat size."""
        if not anchors:
            raise ValueError("no stable people found to build an auto seat map")
        pts = np.asarray(anchors, float)
        hs = np.asarray(heights, float)
        order = np.argsort(-pts[:, 1])  # bottom of image (front) first
        rows_idx: list[list[int]] = [[order[0]]]
        for i in order[1:]:
            prev = rows_idx[-1]
            ref_y = np.mean(pts[prev, 1])
            ref_h = np.mean(hs[prev + [i]])
            if ref_y - pts[i, 1] > 0.35 * ref_h:
                rows_idx.append([i])
            else:
                prev.append(i)

        seats, grid = [], {}
        for r, members in enumerate(rows_idx):
            for c, i in enumerate(sorted(members, key=lambda j: pts[j, 0])):
                others = np.delete(pts, i, axis=0)
                nearest = np.min(np.hypot(*(others - pts[i]).T)) if len(others) else hs[i]
                radius = float(min(0.5 * nearest, 0.4 * hs[i]))
                rx, ry = radius, radius * 0.8
                ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                poly = np.stack([pts[i, 0] + rx * np.cos(ang), pts[i, 1] + ry * np.sin(ang)], 1)
                label = f"{row_label(r)}{c + 1}"
                grid[(r, c)] = label
                seats.append(Seat(label, r, c, poly.tolist(), pts[i].tolist()))

        # neighbours: same row ±1, and the closest-in-x seat (and its sides) in adjacent rows
        for s in seats:
            nb = [grid[k] for k in ((s.row, s.col - 1), (s.row, s.col + 1)) if k in grid]
            for rr in (s.row - 1, s.row + 1):
                row = [t for t in seats if t.row == rr]
                if row:
                    near = min(row, key=lambda t: abs(t.center[0] - s.center[0]))
                    nb += [grid[k] for k in ((rr, near.col - 1), (rr, near.col), (rr, near.col + 1)) if k in grid]
            s.neighbours = nb
        return cls("auto", list(image_size), seats, len(rows_idx),
                   max(len(m) for m in rows_idx))

    # ---------- queries ----------
    def locate(self, pt) -> Seat | None:
        """Seat whose polygon contains pt (nearest centre if several overlap)."""
        p = (float(pt[0]), float(pt[1]))
        hits = [s for s, poly in zip(self.seats, self._polys) if cv2.pointPolygonTest(poly, p, False) >= 0]
        if not hits:
            return None
        return min(hits, key=lambda s: (s.center[0] - p[0]) ** 2 + (s.center[1] - p[1]) ** 2)

    def seat(self, label: str) -> Seat:
        return self._by_label[label]

    def to_plan(self, pts) -> np.ndarray | None:
        """Image px -> plan coords (grid mode only)."""
        if self.H is None:
            return None
        return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), np.asarray(self.H))[:, 0]

    # ---------- io ----------
    def save(self, path: str | Path):
        d = {k: v for k, v in asdict(self).items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "SeatMap":
        d = json.loads(Path(path).read_text())
        d["seats"] = [Seat(**s) for s in d["seats"]]
        return cls(**d)

    # ---------- drawing ----------
    def draw(self, frame: np.ndarray, occupied: set[str] | None = None,
             colors: dict[str, tuple[int, int, int]] | None = None) -> np.ndarray:
        """colors overrides the per-seat colour (e.g. risk level); otherwise green = occupied."""
        occupied, colors = occupied or set(), colors or {}

        def color_of(s):
            return colors.get(s.label) or ((60, 200, 60) if s.label in occupied else (200, 200, 200))

        overlay = frame.copy()
        for s, poly in zip(self.seats, self._polys):
            cv2.fillPoly(overlay, [poly.astype(np.int32)], color_of(s))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        k = max(1.0, frame.shape[1] / 1280)  # readable on 4K too
        for s, poly in zip(self.seats, self._polys):
            color = color_of(s)
            cv2.polylines(frame, [poly.astype(np.int32)], True, color, max(1, int(k)), cv2.LINE_AA)
            cx, cy = map(int, s.center)
            org = (cx - int(10 * k), cy + int(5 * k))
            cv2.putText(frame, s.label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, (0, 0, 0), int(3 * k), cv2.LINE_AA)
            cv2.putText(frame, s.label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, color, max(1, int(k)), cv2.LINE_AA)
        return frame

    def positions(self) -> dict[str, tuple[float, float]]:
        """Seat positions for the Seat Attention Graph: plan coords (grid) or image centres (auto).
        x grows toward image-right in both, matching the image-space yaw sign."""
        if self.mode == "grid":
            return {s.label: (s.col + 0.5, s.row + 0.5) for s in self.seats}
        return {s.label: (float(s.center[0]), float(s.center[1])) for s in self.seats}


def person_anchor(kpts: np.ndarray, conf: np.ndarray, box: np.ndarray, thr: float = 0.4) -> tuple[float, float]:
    """Point that locates a student on the desk plane: hip midpoint, else estimated from shoulders,
    else 70% down the box. Seated hips sit roughly at desk height."""
    v = conf >= thr
    if v[LHIP] and v[RHIP]:
        return float((kpts[LHIP, 0] + kpts[RHIP, 0]) / 2), float((kpts[LHIP, 1] + kpts[RHIP, 1]) / 2)
    if v[LSH] and v[RSH]:
        sw = float(np.hypot(*(kpts[RSH] - kpts[LSH])))
        return float((kpts[LSH, 0] + kpts[RSH, 0]) / 2), float((kpts[LSH, 1] + kpts[RSH, 1]) / 2 + 1.2 * sw)
    return float((box[0] + box[2]) / 2), float(box[1] + 0.7 * (box[3] - box[1]))


def is_standing(kpts: np.ndarray, conf: np.ndarray, box: np.ndarray, thr: float = 0.4) -> bool:
    """Seated: thighs roughly horizontal, knees near hip height. Standing: knees a torso-length lower."""
    f = body_features(kpts, conf, box, thr)
    kd = f["knee_drop"]
    return bool(kd == kd and kd > 0.6 and f["box_aspect"] > 2.0)


class AutoSeatBuilder:
    """Collects per-track anchors over the first frames, then builds an auto SeatMap."""

    def __init__(self, min_presence: float = 0.3):
        self.min_presence = min_presence
        self.frames = 0
        self.tracks: dict[int, list[tuple[float, float, float]]] = {}

    def add(self, people) -> None:
        self.frames += 1
        for p in people:
            if p.track_id is None:
                continue
            ax, ay = person_anchor(p.kpts, p.kpt_conf, p.box)
            self.tracks.setdefault(p.track_id, []).append(
                (ax, ay, float(p.box[3] - p.box[1]), float(is_standing(p.kpts, p.kpt_conf, p.box))))

    def build(self, image_size) -> SeatMap:
        """Raises ValueError if nobody stayed in view long enough (caller should keep collecting)."""
        stable = [np.asarray(v) for v in self.tracks.values() if len(v) >= self.min_presence * self.frames]
        # Drop people who look standing (an invigilator lingering by a desk) — but only among seated
        # students: facing a webcam up close, a seated person's knees also appear below the hips.
        seated = [v for v in stable if np.mean(v[:, 3]) < 0.5]
        if len(seated) >= 2:
            stable = seated
        anchors = [tuple(np.median(v[:, :2], axis=0)) for v in stable]
        heights = [float(np.median(v[:, 2])) for v in stable]
        # merge tracks that are really the same seat (tracker ID switch during calibration)
        keep_a, keep_h = [], []
        for a, h in sorted(zip(anchors, heights), key=lambda t: -t[1]):
            if all(np.hypot(a[0] - b[0], a[1] - b[1]) > 0.25 * h for b in keep_a):
                keep_a.append(a)
                keep_h.append(h)
        return SeatMap.from_anchors(keep_a, keep_h, image_size)
