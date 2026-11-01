"""Tier-1 geometric head pose + torso cues from COCO-17 keypoints (spec §6.5 T1, §7). Cost ≈ 0.

Angles are approximate and in IMAGE terms, not anatomical left/right:
  yaw   > 0  face turned toward image-right,  < 0  toward image-left
  pitch > 0  head up,                         < 0  head down
Absolute values carry a camera-angle bias; the per-seat baseline (P2) turns them into z-scores.

Head model (cm, from head centre): ears ±7.5 sideways, eyes ±3.2 sideways and 7 forward, nose tip
10 forward. Projected, nose offset / half-width ≈ K·tan(yaw), with K = 10/7.5 for the ear pair and
(10-7)/3.2 for the eye pair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH, LWRIST, RWRIST, LHIP, RHIP, LKNEE, RKNEE = 5, 6, 9, 10, 11, 12, 13, 14
K_EAR, K_EYE = 10 / 7.5, 3 / 3.2
PROFILE_YAW = 75.0
RAISED_WRIST_DROP = 0.1  # wrist no lower than 0.1 shoulder-widths below the shoulder line


@dataclass
class HeadPose:
    yaw: float | None
    pitch: float | None
    facing: str      # camera | away | profile | unknown
    source: str      # ears | eyes | profile | none
    head_px: float   # apparent head width in pixels (0 if unknown)
    origin: tuple[float, float] | None  # where to draw the direction arrow from


def head_pose(kpts: np.ndarray, conf: np.ndarray, thr: float = 0.4) -> HeadPose:
    v = conf >= thr
    nose, ears, eyes = v[NOSE], v[LEAR] and v[REAR], v[LEYE] and v[REYE]
    x, y = kpts[:, 0], kpts[:, 1]

    if ears:
        mid = (x[LEAR] + x[REAR]) / 2, (y[LEAR] + y[REAR]) / 2
        half = np.hypot(x[REAR] - x[LEAR], y[REAR] - y[LEAR]) / 2
        if nose and half > 1:
            return _frontal(x[NOSE], y[NOSE], mid, half, K_EAR, "ears", 2 * half)
        if not (nose or v[LEYE] or v[REYE]):  # ears but no face -> back of the head
            return HeadPose(None, None, "away", "none", 2 * half, mid)
    if eyes and nose:
        mid = (x[LEYE] + x[REYE]) / 2, (y[LEYE] + y[REYE]) / 2
        half = np.hypot(x[REYE] - x[LEYE], y[REYE] - y[LEYE]) / 2
        if half > 1:
            return _frontal(x[NOSE], y[NOSE], mid, half, K_EYE, "eyes", 4.7 * half)
    if nose and (v[LEAR] ^ v[REAR]):  # one ear only -> profile; nose side gives the sign
        ear = LEAR if v[LEAR] else REAR
        dx = x[NOSE] - x[ear]
        return HeadPose(math.copysign(PROFILE_YAW, dx), None, "profile", "profile",
                        2 * abs(dx), (float(x[NOSE]), float(y[NOSE])))
    if v[LEAR] or v[REAR]:
        ear = LEAR if v[LEAR] else REAR
        return HeadPose(None, None, "away", "none", 0.0, (float(x[ear]), float(y[ear])))
    return HeadPose(None, None, "unknown", "none", 0.0, None)


def _frontal(nx, ny, mid, half, k, source, head_px) -> HeadPose:
    yaw = math.degrees(math.atan((nx - mid[0]) / half / k))
    # pair width shrinks with cos(yaw): undo that before measuring the vertical offset
    half_true = half / max(math.cos(math.radians(yaw)), 0.3)
    pitch = -math.degrees(math.atan((ny - mid[1]) / half_true / k))
    return HeadPose(yaw, pitch, "camera", source, float(head_px), (float(nx), float(ny)))


def body_features(kpts: np.ndarray, conf: np.ndarray, box: np.ndarray, thr: float = 0.4) -> dict:
    """Torso / posture cues. NaN where the needed keypoints are not visible."""
    v = conf >= thr
    x, y = kpts[:, 0], kpts[:, 1]
    nan = float("nan")
    f = {"shoulder_w": nan, "shoulder_tilt": nan, "lean_x": nan, "torso_len": nan,
         "shoulder_ratio": nan, "knee_drop": nan,
         "box_aspect": float((box[3] - box[1]) / max(box[2] - box[0], 1))}
    if not (v[LSH] and v[RSH]):
        return f
    sw = float(np.hypot(x[RSH] - x[LSH], y[RSH] - y[LSH]))
    f["shoulder_w"] = sw
    # image-space tilt of the shoulder line, folded to [-90, 90]
    tilt = math.degrees(math.atan2(y[LSH] - y[RSH], x[LSH] - x[RSH]))
    f["shoulder_tilt"] = (tilt + 90) % 180 - 90
    if v[LHIP] and v[RHIP] and sw > 1:
        neck = (x[LSH] + x[RSH]) / 2, (y[LSH] + y[RSH]) / 2
        hip = (x[LHIP] + x[RHIP]) / 2, (y[LHIP] + y[RHIP]) / 2
        tl = float(np.hypot(neck[0] - hip[0], neck[1] - hip[1]))
        f["lean_x"] = float((neck[0] - hip[0]) / sw)
        f["torso_len"] = tl
        f["shoulder_ratio"] = sw / max(tl, 1)   # drops when the torso rotates away from camera
        knees = [y[k] for k in (LKNEE, RKNEE) if v[k]]
        if knees and tl > 1:
            f["knee_drop"] = float((np.mean(knees) - hip[1]) / tl)  # large when standing
    return f


def hand_features(kpts: np.ndarray, conf: np.ndarray, thr: float = 0.4) -> dict:
    """Wrist cues, normalised by shoulder width (spec §7). Desk lines are replaced by the per-seat
    baseline: 'hands below desk' = wrists much lower than this student's normal, or hidden.
      wrist_drop  mean visible-wrist y below the shoulder line / shoulder width
      wrists_vis  number of visible wrists (0-2); hands in the lap are usually hidden by the desk
      hand_face   nearest RAISED wrist (at or above shoulder level) to nose / shoulder width
                  (earpiece, covering mouth). Hands resting on the desk are close to the face in pixels
                  in a webcam close-up, so lowered wrists never count.
    """
    v = conf >= thr
    nan = float("nan")
    f = {"wrist_drop": nan, "wrists_vis": float(v[LWRIST] + v[RWRIST]), "hand_face": nan}
    if not (v[LSH] and v[RSH]):
        return f
    sw = float(np.hypot(*(kpts[RSH] - kpts[LSH])))
    if sw < 1:
        return f
    sh_y = (kpts[LSH, 1] + kpts[RSH, 1]) / 2
    wrists = [w for w in (LWRIST, RWRIST) if v[w]]
    if wrists:
        f["wrist_drop"] = float((np.mean(kpts[wrists, 1]) - sh_y) / sw)
        raised = [w for w in wrists if (kpts[w, 1] - sh_y) / sw < RAISED_WRIST_DROP]
        if v[NOSE] and raised:
            f["hand_face"] = float(min(np.hypot(*(kpts[w] - kpts[NOSE])) for w in raised) / sw)
    return f


def pose_features(kpts: np.ndarray, conf: np.ndarray, box: np.ndarray, thr: float = 0.4) -> dict:
    """Flat per-person feature dict (head + body) used for logging and D1 evaluation."""
    hp = head_pose(kpts, conf, thr)
    f = {"yaw": hp.yaw if hp.yaw is not None else float("nan"),
         "pitch": hp.pitch if hp.pitch is not None else float("nan"),
         "facing_camera": float(hp.facing == "camera"), "facing_away": float(hp.facing == "away"),
         "facing_profile": float(hp.facing == "profile"), "head_px": hp.head_px,
         "kp_conf_upper": float(np.mean(conf[:7]))}
    f.update(body_features(kpts, conf, box, thr))
    return f
