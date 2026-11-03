"""Face masking for evidence clips (spec §13: "faces blurred (face region derived from keypoints)").

SAGA never detects or recognises faces, so there is no face box to work from — the region is derived
from the pose keypoints the pipeline already has (nose, eyes, ears), falling back to the top of the
person box when the head is turned away or the head points are missing.

The mask is a **mosaic**, not a Gaussian blur. A Gaussian blur is a convolution and can be partially
inverted; averaging pixels into large blocks throws the information away. Every person in the frame
is masked, not only the seat an alert is about — a bystander has the same claim to privacy as the
student being reviewed.
"""
from __future__ import annotations

import cv2
import numpy as np

NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH = 5, 6
HEAD_KPTS = (NOSE, LEYE, REYE, LEAR, REAR)

MOSAIC_BLOCKS = 6      # the face region is reduced to this many blocks across, then blown back up
MARGIN = 1.45          # grow the head region so hair/chin/ears are covered too


def head_region(person, thr: float = 0.4) -> tuple[int, int, int, int] | None:
    """(x1, y1, x2, y2) around this person's head, in frame pixels. None if nothing to go on."""
    v = person.kpt_conf >= thr
    pts = [person.kpts[i] for i in HEAD_KPTS if v[i]]
    if pts:
        pts = np.asarray(pts, float)
        cx, cy = pts.mean(axis=0)
        spread = float(np.max(np.linalg.norm(pts - [cx, cy], axis=1))) if len(pts) > 1 else 0.0
        if v[LSH] and v[RSH]:
            shoulder = float(np.hypot(*(person.kpts[RSH] - person.kpts[LSH])))
        else:
            shoulder = float(person.box[2] - person.box[0])
        # one visible point (a profile ear, say) tells us nothing about size: use the body scale
        radius = max(spread * MARGIN, 0.42 * shoulder, 10.0)
        return int(cx - radius), int(cy - radius), int(cx + radius), int(cy + radius)
    # head not visible (facing away, occluded): mask the top of the detection box anyway
    x1, y1, x2, y2 = (float(v) for v in person.box)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    h = (y2 - y1) * 0.3
    return int(x1), int(y1), int(x2), int(y1 + h)


def mask_face(img: np.ndarray, box: tuple[int, int, int, int], blocks: int = MOSAIC_BLOCKS) -> None:
    """Mosaic the elliptical face area inside `box`, in place."""
    h, w = img.shape[:2]
    x1, y1 = max(box[0], 0), max(box[1], 0)
    x2, y2 = min(box[2], w), min(box[3], h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(blocks, 1), max(blocks, 1)), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    # Ellipse axes are grown past the box so the corners keep little of the original face; a plain
    # rectangle would be safer still but reads as a censor bar over the whole head and shoulders.
    mask = np.zeros(roi.shape[:2], np.uint8)
    cx, cy = (x2 - x1) // 2, (y2 - y1) // 2
    cv2.ellipse(mask, (cx, cy), (int(cx * 1.12), int(cy * 1.12)), 0, 0, 360, 255, -1)
    roi[mask > 0] = pixelated[mask > 0]


def blur_faces(img: np.ndarray, people: list, scale: float = 1.0, thr: float = 0.4) -> int:
    """Mask every face in the frame. `scale` converts keypoint coordinates (original resolution)
    to this image's resolution. Returns how many faces were masked."""
    n = 0
    for p in people:
        box = head_region(p, thr)
        if box is None:
            continue
        mask_face(img, tuple(int(round(v * scale)) for v in box))
        n += 1
    return n
