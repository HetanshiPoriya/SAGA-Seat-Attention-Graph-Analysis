"""P6 retention job (spec §13: evidence clips "auto-deleted after a configurable retention window").

Deletes evidence clips older than the retention window and marks them deleted in the database, so
an alert that has lost its clip says so instead of pointing at a missing file. Run it on a schedule
(Windows Task Scheduler / cron), or let `run_seats.py --evidence` run it once at startup.

    python scripts\\retention.py --days 30            # delete clips older than 30 days
    python scripts\\retention.py --days 0 --yes       # end of exam: delete every clip
    python scripts\\retention.py --days 30 --dry-run  # show what would go
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS  # noqa: E402
from edge.evidence.clips import purge  # noqa: E402
from reasoner.store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0, help="keep clips younger than this")
    ap.add_argument("--dir", default=str(OUTPUTS / "evidence"))
    ap.add_argument("--db", default=str(OUTPUTS / "saga.db"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation when --days is 0")
    args = ap.parse_args()

    folder = Path(args.dir)
    clips = sorted(folder.glob("clip_*.mp4"))
    cutoff = time.time() - args.days * 86400
    doomed = [p for p in clips if p.stat().st_mtime < cutoff]
    size = sum(p.stat().st_size for p in doomed) / 1e6
    print(f"{len(clips)} clip(s) in {folder}; {len(doomed)} older than {args.days} day(s) ({size:.1f} MB)")
    if not doomed:
        return
    if args.dry_run:
        for p in doomed:
            print(f"  would delete {p.name}")
        return
    if args.days == 0 and not args.yes:
        if input("Delete EVERY evidence clip? type 'delete' to confirm: ").strip() != "delete":
            print("cancelled")
            return

    gone = purge(folder, args.days)
    if Path(args.db).exists():
        store = Store(args.db)
        marked = store.mark_clips_deleted([str(p) for p in gone])
        store.close()
        print(f"deleted {len(gone)} clip(s); {marked} database row(s) marked deleted")
    else:
        print(f"deleted {len(gone)} clip(s) (no database at {args.db})")


if __name__ == "__main__":
    main()
