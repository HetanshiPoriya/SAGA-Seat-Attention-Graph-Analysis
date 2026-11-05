"""Temporal model (spec §9.3 Model A): 2-layer GRU, hidden 64, trained on NORMAL behaviour only.

Why forecasting rather than a classifier: we have no labelled real misconduct, and a classifier
trained on injected behaviour would mostly learn the injector's own shapes. A model that learns to
predict the next feature vector of a normal student needs no misconduct labels at all — it scores a
moment by how surprising it is. That is the "normal-only pretraining" branch of §9.3; the MIL
fine-tune needs session-level labels from real recordings we do not have yet.

The score is deliberately *not* an alert. It is fused into the rule layer (reasoner/temporal.py):
a rule event is required, and the surprise either strengthens or damps it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.temporal.features import N_FEATURES, WINDOW


class Forecaster(nn.Module):
    """Given the first `window-1` frames, predict the last one."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])


def prediction_error(model: Forecaster, batch: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Per-window error between the predicted and the actual last frame.

    Only the *values* of features that are actually present are compared — scoring a model for
    mispredicting a wrist that was never visible would be noise, not surprise.
    """
    x, target = batch[:, :-1], batch[:, -1]
    pred = model(x)
    half = target.shape[1] // 2
    present = target[:, half:]
    err = (pred[:, :half] - target[:, :half]) ** 2
    if weights is not None:
        err = err * weights
    return (err * present).sum(dim=1) / present.sum(dim=1).clamp(min=1.0)


@dataclass
class Calibration:
    """Turns a raw error into a 0–1 surprise, using the error distribution of NORMAL data."""
    median: float = 0.0
    p95: float = 1.0
    p99: float = 1.0
    feature_std: list[float] = field(default_factory=list)

    def surprise(self, err: np.ndarray | float) -> np.ndarray | float:
        """0 at the normal median, 1 at the normal 99th percentile, >1 beyond it."""
        span = max(self.p99 - self.median, 1e-6)
        return np.clip((np.asarray(err, dtype=float) - self.median) / span, 0.0, 3.0)


class TemporalScorer:
    """Loads the trained model and scores windows. Used live by reasoner/temporal.py."""

    def __init__(self, path: str | Path, device: str = "cpu"):
        path = Path(path)
        blob = torch.load(path, map_location=device, weights_only=False)
        self.model = Forecaster(**blob["arch"])
        self.model.load_state_dict(blob["state"])
        self.model.eval()
        self.calib = Calibration(**blob["calibration"])
        self.window = blob.get("window", WINDOW)
        self.device = device
        self.weights = torch.tensor(blob["feature_weights"], dtype=torch.float32) \
            if blob.get("feature_weights") else None

    @torch.no_grad()
    def score(self, window_batch: np.ndarray) -> np.ndarray:
        """(N, window, F) -> surprise per window."""
        if len(window_batch) == 0:
            return np.zeros(0)
        x = torch.as_tensor(np.asarray(window_batch, dtype=np.float32), device=self.device)
        err = prediction_error(self.model, x, self.weights).cpu().numpy()
        return np.asarray(self.calib.surprise(err))


def save(model: Forecaster, calib: Calibration, path: str | Path, arch: dict,
         feature_weights: list[float] | None = None, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "arch": arch, "window": WINDOW,
                "calibration": calib.__dict__, "feature_weights": feature_weights,
                "meta": meta or {}}, path)
    (path.with_suffix(".json")).write_text(json.dumps({"arch": arch, "calibration": calib.__dict__,
                                                       "meta": meta or {}}, indent=2), encoding="utf-8")
    return path
