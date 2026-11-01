"""Overlay drawing shared by the scripts. Sizes scale with the frame, so 4K video stays readable."""
import math

import cv2
import numpy as np

from edge.perception.pose_tracker import SKELETON

FONT = cv2.FONT_HERSHEY_SIMPLEX


def ui_scale(frame) -> float:
    """1.0 up to 1280 px wide, proportionally larger above (3.0 for 4K)."""
    return max(1.0, frame.shape[1] / 1280)


def hud(frame, lines):
    s = ui_scale(frame)
    lh, th = int(24 * s), max(1, int(s))
    w = max(cv2.getTextSize(t, FONT, 0.6 * s, th)[0][0] for t in lines)
    cv2.rectangle(frame, (0, 0), (w + int(20 * s), lh * len(lines) + int(8 * s)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (int(10 * s), lh * (i + 1)), FONT, 0.6 * s, (255, 255, 255), th, cv2.LINE_AA)


def label(frame, text, org, color, scale=0.5):
    s = ui_scale(frame)
    scale, th = scale * s, max(1, int(s))
    (tw, tht), _ = cv2.getTextSize(text, FONT, scale, th)
    x, y = int(org[0]), max(int(org[1]), tht + 6)
    pad = int(3 * s)
    cv2.rectangle(frame, (x, y - tht - 2 * pad), (x + tw + 2 * pad, y), color, -1)
    cv2.putText(frame, text, (x + pad, y - pad), FONT, scale, (0, 0, 0), th, cv2.LINE_AA)


def draw_skeleton(frame, person, color, kpt_thresh=0.4):
    s = ui_scale(frame)
    ok = person.kpt_conf >= kpt_thresh
    pts = person.kpts.astype(int)
    for a, b in SKELETON:
        if ok[a] and ok[b]:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, max(2, int(2 * s)), cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        cv2.circle(frame, tuple(pts[i]), int(3 * s), color, -1, cv2.LINE_AA)


def draw_head_arrow(frame, origin, yaw, pitch, head_px, color=(0, 0, 255)):
    """Arrow in image space: right for +yaw, up for +pitch."""
    if origin is None or yaw is None:
        return
    L = max(head_px, 12) * 1.6
    dx = L * math.sin(math.radians(yaw))
    dy = -L * math.sin(math.radians(pitch or 0.0))
    o = (int(origin[0]), int(origin[1]))
    cv2.arrowedLine(frame, o, (int(o[0] + dx), int(o[1] + dy)), color, max(2, int(2 * ui_scale(frame))),
                    cv2.LINE_AA, tipLength=0.3)


def draw_sag_edges(frame, centres: dict, edges: dict, color=(0, 140, 255)):
    """Seat Attention Graph arrows between seat centres; thickness grows with the number of glances."""
    s = ui_scale(frame)
    for (src, dst), (n, _) in edges.items():
        if src not in centres or dst not in centres:
            continue
        a, b = np.array(centres[src], float), np.array(centres[dst], float)
        d = b - a
        if np.hypot(*d) < 1:
            continue
        a2, b2 = a + 0.2 * d, a + 0.8 * d  # keep clear of the seat labels
        normal = np.array([-d[1], d[0]]) / np.hypot(*d) * 6 * s  # offset so i->j and j->i don't overlap
        cv2.arrowedLine(frame, tuple((a2 + normal).astype(int)), tuple((b2 + normal).astype(int)), color,
                        max(2, int(min(2 + n, 8) * s)), cv2.LINE_AA, tipLength=0.15)
        mid = (a2 + b2) / 2 + normal * 2
        label(frame, f"{n}x", mid, color, 0.45)
