# SAGA — Seat Attention Graph Analysis

Privacy-first, relation-aware AI invigilation. Full design: [SAGA_PROJECT_SPEC.md](SAGA_PROJECT_SPEC.md).

## Setup (Windows, PowerShell, Python 3.13)

```powershell
cd C:\NirikshakAI
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

Datasets live **outside** the repo in `C:\nirikshak_data\` (override with `$env:SAGA_DATA`).

## P-1 — Data

```powershell
python scripts\inspect_d1.py                                        # Kaggle copy (images only)
python scripts\inspect_d1.py --root C:\nirikshak_data\smart_invigilator   # labelled Roboflow copy
```

## P0 — Pose + tracking

```powershell
python scripts\run_pose.py                                  # webcam
python scripts\run_pose.py --source C:\path\to\clip.mp4 --save
```

Keys: `q`/`Esc` quit · `space` pause · `b` toggle boxes.

## P1 — Seats

```powershell
# 1. calibrate: click FRONT-LEFT, FRONT-RIGHT, BACK-RIGHT, BACK-LEFT of the desk area
python scripts\calibrate.py --source 0 --rows 1 --cols 3            # -> outputs\calib\webcam.json
# 2. run with seat locking + per-seat head pose (+ features CSV)
python scripts\run_seats.py --source 0 --calib outputs\calib\webcam.json

# no clicking: pseudo seat map from the first N frames (for classroom videos)
python scripts\run_seats.py --source clip.mp4 --auto-seats 60 --save

# D1 posture evaluation (needs the labelled dataset)
python scripts\eval_d1.py --root C:\nirikshak_data\smart_invigilator   # -> outputs\d1_eval\report.md
```

Head pose is in image terms: `y+` = face turned to image-right, `p-` = looking down.

## P2 — Baseline, rules, risk

```powershell
# live: sit normally during the baseline, then try glancing sideways / hands under the desk
python scripts\run_seats.py --source 0 --auto-seats 30 --reason --calib-seconds 30

# replay a saved recording through the reasoner (try other rules/config without re-recording)
python scripts\replay.py --features outputs\features_webcam_....csv --calib-seconds 60

# evaluation by behaviour injection -> outputs\p2_eval\report.md + sweep.png
python scripts\simulate.py                                   # synthetic class (pipeline test)
python scripts\simulate.py --features outputs\features_....csv  # REAL normal recording (≥ 6 min)
```

Rules are plain YAML in [reasoner/rules/rules.yaml](reasoner/rules/rules.yaml); thresholds and the
baseline length in [reasoner/config.yaml](reasoner/config.yaml). Seat colours: grey-white =
calibrating, green / amber / red = risk level, dark grey = coverage gap. Events and alerts are
stored in `outputs\saga.db` (SQLite; same tables as the planned Postgres schema).

## P3 — Dashboard

```powershell
python scripts\dashboard.py --demo                       # scripted 15-seat class, no camera needed
python scripts\run_seats.py --source 0 --auto-seats 30 --dashboard --calib-seconds 30   # live webcam
python scripts\run_seats.py --source C:\nirikshak_data\videos\clip.mp4 --auto-seats 45 --dashboard
```

Open http://127.0.0.1:8000 — hall map (risk colours, Seat Attention Graph arrows), alert feed with
reasons and Confirm / Dismiss / Unsure (stored for recalibration), per-seat timeline against the
student's own baseline band. Local-only (127.0.0.1); single static page, no build step (the spec's
React stack can replace it later without changing the API: `/api/state`, `/api/alerts`,
`/api/alerts/{id}/feedback`, `/api/seats/{seat}/timeline`, `/api/health`, `/ws`).

## P4 — Seat Attention Graph

Built into the reasoner ([reasoner/sag.py](reasoner/sag.py), settings under `sag:` in
[reasoner/config.yaml](reasoner/config.yaml)). Desk-level glances become directed edges
`seat → neighbour on that side`; patterns: `COPYING_ONE_WAY` (i → j repeatedly, j not looking
back), `MUTUAL_ORIENTATION` (i ⇄ j within 5 s, twice), `ATTENTION_HUB` (≥ 3 seats toward one
seat; informational). Evaluate with `python scripts\simulate.py` (tiers: synthetic, synthetic +
real detector jitter, real-track bootstrap stress test; with vs without SAG).

Tests: `python -m pytest -q`

## Layout

```
config.py                       paths (data root outside repo)
edge/ingest/sources.py          open webcam/video, model weights location
edge/perception/                pose_tracker.py (YOLO pose + BoT-SORT), botsort.yaml,
                                head_pose.py (geometric yaw/pitch + torso features)
edge/calibration/               seatmap.py (grid homography / auto seats), seat_lock.py
edge/viz.py                     overlay drawing
reasoner/                       baseline.py (per-seat median/MAD, rolling recent direction),
                                rules/ (safe YAML rule engine), sag.py (Seat Attention Graph),
                                risk.py, store.py (SQLite + feedback), reasoner.py (+ despike filter)
api/                            live.py (shared hall state), app.py (FastAPI + WebSocket), static/index.html
ml/datasets/cctv_exam.py        D1 loader (auto-detects YOLO / COCO / CSV / folder layouts)
ml/injection/                   session.py (real CSV / synthetic class), inject.py (behaviours + ground truth),
                                bootstrap.py (real-track bootstrap, real detector jitter)
ml/eval/                        d1_posture.py, injection_eval.py (recall, FA/student-hour, sweeps)
scripts/                        inspect_d1, run_pose, calibrate, run_seats, eval_d1, replay, simulate, dashboard
tests/                          core regression tests (pytest)
models/  outputs/               weights / generated files (gitignored)
```
