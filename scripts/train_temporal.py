"""P7: train the temporal model on NORMAL behaviour only (spec §9.3 normal-only pretraining).

    python scripts\\train_temporal.py                       # synthetic + real detector jitter
    python scripts\\train_temporal.py --features outputs\\features_webcam_*.csv   # your own recording
    python scripts\\train_temporal.py --epochs 40 --sessions 12

No misconduct labels are used or needed: the model learns to predict the next feature vector of a
normal student, and anything it cannot predict is "surprising". Training sessions are held out from
the evaluation seeds in scripts/simulate.py, so the reported false-alarm reduction is not measured
on the data the model was fitted to.

Writes models\\temporal.pt (+ .json with the calibration) and outputs\\p7_eval\\training.md.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS, OUTPUTS  # noqa: E402
from ml.injection.bootstrap import add_real_jitter, jitter_pool, load_tracks  # noqa: E402
from ml.injection.session import from_features_csv, synthetic_session  # noqa: E402
from ml.temporal.features import N_FEATURES, WINDOW, session_sequences, windows  # noqa: E402
from ml.temporal.model import Calibration, Forecaster, prediction_error, save  # noqa: E402

TRAIN_SEEDS = range(100, 200)   # evaluation uses seeds 0..n, so training never sees them


def build_windows(args, calib_s: float) -> tuple[np.ndarray, list[str]]:
    notes = []
    if args.features:
        paths = sorted(glob.glob(args.features))
        sessions = [from_features_csv(p) for p in paths]
        notes.append(f"{len(sessions)} real recording(s): {', '.join(Path(p).name for p in paths)}")
    else:
        real_csvs = [c for c in glob.glob(str(OUTPUTS / "features_*.csv")) if "test_clip" not in c]
        pool = jitter_pool(load_tracks(real_csvs)) if real_csvs else None
        sessions = []
        for i, seed in enumerate(list(TRAIN_SEEDS)[:args.sessions]):
            s = synthetic_session(rows=3, cols=5, duration_s=args.duration, seed=seed)
            if pool:
                add_real_jitter(s, pool, seed)
            sessions.append(s)
        notes.append(f"{len(sessions)} synthetic normal sessions (seeds {min(TRAIN_SEEDS)}+)"
                     + (f" + real detector jitter from {len(real_csvs)} clips" if pool else ""))

    chunks = []
    for s in sessions:
        for seat, seq in session_sequences(s, calib_s).items():
            w = windows(seq, WINDOW, stride=args.stride)
            if len(w):
                chunks.append(w)
    if not chunks:
        sys.exit("no usable windows — are the recordings longer than the baseline?")
    return np.concatenate(chunks), notes


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", help="glob of real features CSVs to train on instead of synthetic")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--duration", type=float, default=900.0)
    ap.add_argument("--calib", type=float, default=180.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--stride", type=int, default=2, help="window stride (1 = every frame)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default=str(MODELS / "temporal.pt"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    data, notes = build_windows(args, args.calib)
    rng = np.random.default_rng(0)
    rng.shuffle(data)
    n_val = int(len(data) * args.val_frac)
    val, train = data[:n_val], data[n_val:]
    print(f"{len(train)} training windows, {len(val)} validation ({N_FEATURES} features x {WINDOW} frames)")

    # Features vary in how noisy they are; weight the loss by 1/variance so the model is not
    # dominated by the jumpiest channel (yaw_z on a profile seat, say).
    values = train[:, -1, :N_FEATURES // 2]
    std = values.std(axis=0)
    weights = torch.tensor(1.0 / np.clip(std, 0.1, None) ** 2, dtype=torch.float32)
    weights = weights / weights.mean()

    arch = {"n_features": N_FEATURES, "hidden": args.hidden, "layers": args.layers}
    model = Forecaster(**arch)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_t = torch.as_tensor(train)
    val_t = torch.as_tensor(val)

    best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_t))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            batch = train_t[perm[i:i + args.batch]]
            loss = prediction_error(model, batch, weights).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(batch)
        sched.step()
        model.eval()
        with torch.no_grad():
            v = float(prediction_error(model, val_t, weights).mean()) if n_val else float("nan")
        print(f"  epoch {epoch + 1:3}/{args.epochs}  train {total / len(train_t):.4f}  val {v:.4f}")
        if v == v and v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    # Calibrate on normal data: the surprise scale is defined by what normal already looks like.
    model.eval()
    with torch.no_grad():
        errs = prediction_error(model, val_t if n_val else train_t, weights).numpy()
    calib = Calibration(median=float(np.median(errs)), p95=float(np.percentile(errs, 95)),
                        p99=float(np.percentile(errs, 99)), feature_std=std.tolist())
    meta = {"notes": notes, "windows": int(len(data)), "epochs": args.epochs,
            "val_error": best, "trained_at": time.strftime("%Y-%m-%d %H:%M")}
    path = save(model, calib, args.out, arch, weights.tolist(), meta)

    out = Path(OUTPUTS / "p7_eval")
    out.mkdir(parents=True, exist_ok=True)
    md = ["# P7 temporal model — training\n", *(f"- {n}" for n in notes),
          f"- {len(data)} windows of {WINDOW} frames x {N_FEATURES} features",
          f"- 2-layer GRU, hidden {args.hidden}, {args.epochs} epochs, best val error {best:.4f}",
          f"- normal-error calibration: median {calib.median:.4f}, p95 {calib.p95:.4f}, p99 {calib.p99:.4f}",
          f"- trained in {time.perf_counter() - t0:.0f}s -> `{path}`\n",
          "Surprise = (error - median) / (p99 - median), clipped to [0, 3]: 0 is typical normal "
          "behaviour, 1 is the 99th percentile of normal.\n"]
    (out / "training.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
