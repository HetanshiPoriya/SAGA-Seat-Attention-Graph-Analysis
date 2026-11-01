"""Project-wide paths. Datasets live OUTSIDE the repo; override with SAGA_DATA."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("SAGA_DATA") or os.environ.get("NIRIKSHAK_DATA") or r"C:\nirikshak_data")

D1_CCTV_EXAM = DATA_ROOT / "CCTV-Exam-Monitor-Dataset"

OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"
