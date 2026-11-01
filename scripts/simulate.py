"""P2/P4: evaluate the reasoner + Seat Attention Graph by behaviour injection (spec §16.3)
-> outputs\\p2_eval\\report.md

Synthetic class, synthetic + real detector jitter, and a real-track bootstrap stress test
(real tracks = features CSVs from run_seats.py on real clips, outputs\\features_*.csv):
    python scripts\\simulate.py
    python scripts\\simulate.py --quick
Real normal data (a features CSV from run_seats.py, e.g. 10 min of you working normally):
    python scripts\\simulate.py --features outputs\\features_webcam_....csv [--seatmap outputs\\calib\\x.json]
"""
import argparse
import csv
import glob
import math
import warnings
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS  # noqa: E402
from ml.eval.injection_eval import per_type, run, scenario, summarize, sweep  # noqa: E402
from ml.injection.bootstrap import (add_real_jitter, bootstrap_session, jitter_pool, load_tracks,  # noqa: E402
                                    noise_stats)
from ml.injection.session import from_features_csv, synthetic_session  # noqa: E402

SWEEPS = {"amplitude": [10, 15, 20, 25, 35, 45, 60],
          "hold": [0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
          "k": [1, 2, 3, 4, 6, 8]}
LABELS = {"amplitude": "glance amplitude (°)", "hold": "glance hold (s)", "k": "glances per 2 min"}


def f(v, spec=".2f"):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def table(rows: list[dict], first: str, relational: bool = False) -> str:
    cols = [first, "behaviours", "recall", "alert_recall", "precision", "fa_per_student_hour",
            "false_alerts_per_student_hour"]
    if relational:
        cols[4:4] = ["pair_recall", "attribution", "reversed"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) if c in (first, "behaviours") else f(r[c]) for c in cols) + " |")
    return "\n".join(out)


def plot(sweeps: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5 * len(sweeps), 3.6), sharey=True)
    for ax, (param, rows) in zip(np.atleast_1d(axes), sweeps.items()):
        x = [r[param] for r in rows]
        ax.plot(x, [r["recall"] for r in rows], "o-", label="event detected")
        ax.plot(x, [r["alert_recall"] for r in rows], "s--", label="alert raised (amber/red)")
        ax.set_xlabel(LABELS[param])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("detection rate")
    np.atleast_1d(axes)[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console defaults to cp1252
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices in real tracks
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", help="real features CSV as the normal session (default: synthetic class)")
    ap.add_argument("--seatmap", help="seat map JSON matching the features CSV (seat positions)")
    ap.add_argument("--seeds", type=int, default=6, help="synthetic sessions for the scenario test")
    ap.add_argument("--trials", type=int, default=4, help="sessions per sweep point")
    ap.add_argument("--calib", type=float, default=None, help="baseline seconds (default 180, or 40%% of a real clip)")
    ap.add_argument("--quick", action="store_true", help="fewer seeds/trials, for a fast check")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--real-tracks", default=str(OUTPUTS / "features_*.csv"),
                    help="features CSVs from real clips, for the real-jitter and bootstrap tiers")
    ap.add_argument("--out", default=str(OUTPUTS / "p2_eval"))
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.trials = 2, 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t_start = time.perf_counter()
    md = []

    if args.features:
        base = from_features_csv(args.features, args.seatmap)
        calib = args.calib or min(180.0, 0.4 * base.duration)
        if base.duration - calib < 150:
            sys.exit(f"Recording is {base.duration:.0f}s; need at least {calib + 150:.0f}s "
                     f"(baseline {calib:.0f}s + 150s for injected behaviour). Record longer.")
        md.append(f"# P2 injection evaluation — REAL normal data\n\nsource `{base.source}` · "
                  f"{len(base.seats)} seat(s) · {base.duration / 60:.1f} min · baseline {calib:.0f}s\n")
        untouched = run(base.copy(), [], calib)
        hours = len(base.seats) * (base.duration - calib) / 3600
        md.append(f"## Untouched real seats\n\n{len(untouched.events)} events, {len(untouched.alerts)} alerts in "
                  f"{hours:.2f} student-hours → **{len(untouched.alerts) / hours:.2f} alerts / student-hour**\n")
        for e in untouched.events:
            md.append(f"- t={e.t:.0f}s seat {e.seat}: {e.message}")
        factory = lambda tr: base.copy()  # noqa: E731
    else:
        calib = args.calib or 180.0
        real_csvs = [c for c in glob.glob(args.real_tracks) if "test_clip" not in c]
        tracks = load_tracks(real_csvs) if real_csvs else []
        pool = jitter_pool(tracks) if tracks else None
        md.append("# P2/P4 injection evaluation\n\n"
                  "Three tiers of 'normal' data, from most controlled to most real:\n\n"
                  "1. **synthetic** - generated exam class (AR head noise, posture drift, natural distractors: "
                  "a glance every ~3 min, pairs every ~10 min, assumed level-or-upward).\n"
                  "2. **synthetic + real jitter** - the same behaviour plus the high-frequency head-angle noise and "
                  f"dropouts measured on {len(tracks)} real seat tracks from {len(real_csvs)} clips. Tests robustness "
                  "to real keypoint noise. **Headline tier.**\n"
                  "3. **real-track bootstrap (stress test)** - 15-min sessions chained from 3 s blocks of the real "
                  "tracks. Stock-footage actors look around far more than exam students, and chaining repeats each "
                  "clip's head turns ~60x, so this is pessimistic for false alarms.\n\n"
                  "Real *exam* recordings (SAV classroom video, or a 10-min webcam session via `--features`) are "
                  "still needed for a true false-alarm rate.\n")
        tiers = [("synthetic", False)] + ([("synthetic + real jitter", True)] if pool else [])
        rows, per_beh = [], {}
        for name, jit in tiers:
            for sag in (False, True):
                results = []
                for seed in range(args.seeds):
                    sess = synthetic_session(rows=3, cols=5, duration_s=900, seed=seed)
                    if jit:
                        add_real_jitter(sess, pool, seed)
                    results.append(run(sess, scenario(sess, np.random.default_rng(seed), calib), calib, sag=sag))
                rows.append({"tier": f"{name}, SAG {'on' if sag else 'off'}", **summarize(results)})
                if sag:
                    per_beh[name] = results
        md.append(f"## Mixed scenario ({args.seeds} sessions x 15 seats x 15 min; one of each behaviour per "
                  "session, other seats untouched)\n\n" + table(rows, "tier", relational=True) + "\n")
        md.append("pair_recall = injected copying/collusion attributed to the correct partner seat; "
                  "attribution = share of relational detections naming the right partner; reversed = copier and "
                  "source swapped.\n")
        head = list(per_beh)[-1]
        by_type: dict[str, list] = {}
        for r in per_beh[head]:
            for g, d, a in zip(r.gt, r.detected, r.alerted):
                by_type.setdefault(g["type"], []).append((d, a))
        md.append(f"### Per behaviour ({head}, SAG on)\n\n| behaviour | n | detected | alert raised |\n|---|---|---|---|")
        for k, v in by_type.items():
            md.append(f"| {k} | {len(v)} | {np.mean([d for d, _ in v]):.2f} | {np.mean([a for _, a in v]):.2f} |")
        md.append("")
        if tracks:
            st = noise_stats(tracks)
            md.append("## Real detector noise vs the synthetic assumption\n\n"
                      "| measure | real median | real p90 |\n|---|---|---|")
            for k in ("yaw_mad", "yaw_step_sd", "facing_away", "facing_profile"):
                v = np.array([x[k] for x in st], float)
                md.append(f"| {k} | {np.nanmedian(v):.2f} | {np.nanpercentile(v, 90):.2f} |")
            md.append("")
            boot = [run(bootstrap_session(tracks, seed=seed), [], calib) for seed in range(max(2, args.seeds // 2))]
            b = summarize(boot)
            md.append(f"## Real-track bootstrap - stress test (untouched)\n\n{f(b['fa_per_student_hour'])} events and "
                      f"**{f(b['false_alerts_per_student_hour'])} alerts per student-hour**. Inspection of the source "
                      "clips shows smooth, deliberate head turns by the actors (not detector noise); the rules flag "
                      "them by design. Not representative of a real exam.\n")
        base = None
        factory = ((lambda tr: add_real_jitter(synthetic_session(duration_s=480, seed=tr), pool, tr))  # noqa: E731
                   if pool else None)

    if args.features:
        res = per_type(base, rng, calib, repeats=3)
        md.append("\n## Each behaviour injected alone into the real recording (3× each)\n")
        md.append(table([{"behaviour": k, **summarize(v)} for k, v in res.items()], "behaviour") + "\n")

    if not args.no_sweep:
        md.append("\n## Subtlety sweep — one-way copying (spec §16.3 step 4)\n\n"
                  "Defaults: amplitude 35°, hold 1.5 s, 4 glances per 2 min; one parameter varied at a time.\n")
        sweeps = {}
        for param, values in SWEEPS.items():
            sweeps[param] = sweep(param, values, trials=args.trials, calib_s=calib, session_factory=factory)
            md.append(f"### {LABELS[param]}\n\n" + table(sweeps[param], param) + "\n")
            with open(out / f"sweep_{param}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, list(sweeps[param][0]))
                w.writeheader()
                w.writerows(sweeps[param])
        plot(sweeps, out / "sweep.png", "Detection vs subtlety — " + (
            "real recording" if args.features else "synthetic + real jitter" if factory else "synthetic"))
        md.append("![sweep](sweep.png)\n")

    md.append(f"\n_Generated in {time.perf_counter() - t_start:.0f}s._\n")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nreport -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
