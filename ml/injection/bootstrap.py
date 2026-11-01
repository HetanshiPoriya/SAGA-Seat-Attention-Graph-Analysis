"""Long "real-noise" sessions from short real clips, by block bootstrap.

Real stock clips are 8–22 s, far shorter than a 3-minute baseline. Each real seat track is resampled
to 10 Hz, cut into short blocks, and blocks are chained at random into a long series. Head angles are
re-centred per block (block median -> the track's median) so the joins do not create fake glances.
Everything inside a block — detector jitter, keypoint dropouts, profile views, students facing away,
real fidgeting — is real; only the long-range ordering is synthetic. Report results as
"real-track bootstrap", between the synthetic class and fully real long recordings.
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np

from ml.injection.session import SERIES, SeatSeries, Session, from_features_csv

RECENTRE = ("yaw", "pitch")


def load_tracks(csv_paths, fps=10.0, min_s=5.0, min_present=0.6) -> list[dict]:
    """Real per-seat tracks, resampled to `fps` (nearest frame), keeping seats seen ≥ min_present."""
    tracks = []
    for p in csv_paths:
        sess = from_features_csv(p)
        for seat, s in sess.seats.items():
            if len(s.t) < 2:
                continue
            grid = np.arange(s.t[0], s.t[-1], 1 / fps)
            if len(grid) < min_s * fps:
                continue
            idx = np.clip(np.searchsorted(s.t, grid), 0, len(s.t) - 1)
            present = np.abs(s.t[idx] - grid) < 0.5 / fps * 2
            if present.mean() < min_present:
                continue
            tracks.append({"source": f"{Path(p).stem}:{seat}", "facing": s.facing[idx][present],
                           **{k: s.data[k][idx][present] for k in SERIES}})
    return tracks


def noise_stats(tracks: list[dict]) -> list[dict]:
    """Per-track real noise, to compare with the synthetic generator's assumptions."""
    out = []
    for tr in tracks:
        y = tr["yaw"]
        ok = ~np.isnan(y)
        dy = np.diff(y[ok]) if ok.sum() > 2 else np.array([np.nan])
        out.append({"track": tr["source"], "frames": len(y), "yaw_valid": ok.mean(),
                    "yaw_median": float(np.nanmedian(y)) if ok.any() else float("nan"),
                    "yaw_mad": float(1.4826 * np.nanmedian(np.abs(y - np.nanmedian(y)))) if ok.any() else float("nan"),
                    "yaw_step_sd": float(np.nanstd(dy)),
                    "facing_away": float(np.mean(tr["facing"] == "away")),
                    "facing_profile": float(np.mean(tr["facing"] == "profile")),
                    "wrists_vis": float(np.nanmean(tr["wrists_vis"]))})
    return out


def jitter_pool(tracks: list[dict], fps=10.0, window_s=1.0) -> dict[str, list[np.ndarray]]:
    """High-frequency part of real head angles (value − its 1 s rolling median): detector jitter
    and micro-movement, without the actors' deliberate head turns."""
    w = max(int(window_s * fps) | 1, 3)
    pool = {k: [] for k in RECENTRE}
    for tr in tracks:
        for k in RECENTRE:
            x = tr[k].astype(float)
            if np.isnan(x).mean() > 0.5 or len(x) < 3 * w:
                continue
            pad = np.pad(x, w // 2, mode="edge")
            med = np.array([np.nanmedian(pad[i:i + w]) for i in range(len(x))])
            pool[k].append(x - med)  # NaN where the detector gave no angle
    return pool


def add_real_jitter(sess: Session, pool: dict[str, list[np.ndarray]], seed=0, block_s=3.0, fps=10.0) -> Session:
    """Synthetic behaviour + real detector jitter (and real dropout pattern), in place."""
    rng = np.random.default_rng(seed)
    b = int(block_s * fps)
    for s in sess.seats.values():
        n = len(s.t)
        for k in RECENTRE:
            if not pool[k]:
                continue
            parts, got = [], 0
            while got < n:
                src = pool[k][rng.integers(len(pool[k]))]
                s0 = int(rng.integers(0, max(len(src) - b, 1)))
                seg = src[s0:s0 + b]
                parts.append(seg)
                got += len(seg)
            s.data[k] = s.data[k] + np.concatenate(parts)[:n]  # NaN propagates = real missing angles
    sess.source += " + real jitter"
    return sess


def bootstrap_session(tracks: list[dict], rows=3, cols=5, duration_s=900.0, fps=10.0, block_s=3.0,
                      seed=0) -> Session:
    rng = np.random.default_rng(seed)
    n, b = int(duration_s * fps), int(block_s * fps)
    t = np.arange(n) / fps
    seats, pos = {}, {}
    for r in range(rows):
        for c in range(cols):
            tr = tracks[rng.integers(len(tracks))]
            L = len(tr["yaw"])
            med = {k: np.nanmedian(tr[k]) if np.any(~np.isnan(tr[k])) else np.nan for k in RECENTRE}
            parts = {k: [] for k in SERIES}
            facing = []
            got = 0
            while got < n:
                s0 = int(rng.integers(0, max(L - b, 1)))
                seg = slice(s0, min(s0 + b, L))
                for k in SERIES:
                    x = tr[k][seg].astype(float).copy()
                    if k in RECENTRE and np.any(~np.isnan(x)):
                        x += med[k] - np.nanmedian(x)
                    parts[k].append(x)
                facing.append(tr["facing"][seg])
                got += seg.stop - seg.start
            label = f"{string.ascii_uppercase[r]}{c + 1}"
            seats[label] = SeatSeries(t.copy(), {k: np.concatenate(v)[:n] for k, v in parts.items()},
                                      np.concatenate(facing)[:n].astype(object))
            pos[label] = (float(c + 1), float(r))
    return Session(f"bootstrap(seed={seed}, {len(tracks)} real tracks)", seats, pos)
