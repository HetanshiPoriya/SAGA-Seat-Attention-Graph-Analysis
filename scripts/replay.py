"""P2: re-run the reasoner on a saved features CSV (record once, try different rules/config).

    python scripts\\replay.py --features outputs\\features_webcam_....csv --calib-seconds 60
Prints the event/alert timeline and stores it in outputs\\saga.db.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS  # noqa: E402
from reasoner.reasoner import Reasoner, read_features_csv  # noqa: E402
from reasoner.store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--calib-seconds", type=float, default=None)
    ap.add_argument("--rules", default=None, help="alternative rules YAML")
    ap.add_argument("--db", default=str(OUTPUTS / "saga.db"))
    args = ap.parse_args()

    frames = read_features_csv(args.features)
    if not frames:
        sys.exit("No rows in the features file.")
    r = Reasoner(calib_s=args.calib_seconds, rules_file=args.rules)
    store = Store(args.db)
    sid = store.new_session(f"replay:{Path(args.features).name}", args.features, r.config)
    n_ev = n_al = 0
    seats = set()
    for t, rows in frames:
        seats.update(row["seat"] for row in rows)
        out = r.step(t, rows)
        for e in out.events:
            store.add_event(sid, e)
            n_ev += 1
            print(f"  t={t:7.1f}s  {e.seat:<4} event  {e.rule_id:<20} {e.message}")
        for a in out.alerts:
            store.add_alert(sid, a)
            n_al += 1
            print(f"  t={t:7.1f}s  {a.seat:<4} ALERT  {a.level.upper():<6} risk {a.risk:.2f}  obs {a.observability:.2f}")
    store.close()
    dur = frames[-1][0] - frames[0][0]
    print(f"\n{len(seats)} seats · {dur / 60:.1f} min · baseline {r.baseline.calib_s:.0f}s · "
          f"{n_ev} events · {n_al} alerts  -> {args.db} (session {sid[:8]})")
    if dur <= r.baseline.calib_s:
        print("Note: the recording is not longer than the baseline period, so nothing could be flagged.")


if __name__ == "__main__":
    main()
