"""Project-wide paths. Datasets live OUTSIDE the repo; override with SAGA_DATA."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _data_root() -> Path:
    """Data folder was renamed to C:\\SAGA_data on 2026-11-01; the old name still works."""
    env = os.environ.get("SAGA_DATA") or os.environ.get("NIRIKSHAK_DATA")
    if env:
        return Path(env)
    for candidate in (r"C:\SAGA_data", r"C:\nirikshak_data"):
        if Path(candidate).exists():
            return Path(candidate)
    return Path(r"C:\SAGA_data")


DATA_ROOT = _data_root()

D1_CCTV_EXAM = DATA_ROOT / "CCTV-Exam-Monitor-Dataset"
VIDEOS = DATA_ROOT / "videos"

OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"
