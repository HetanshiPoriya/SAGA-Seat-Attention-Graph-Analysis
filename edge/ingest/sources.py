"""Opening video sources and locating model weights."""
import sys
from pathlib import Path

import cv2

from config import MODELS


def open_source(src: str) -> cv2.VideoCapture:
    if src.isdigit():  # webcam; DirectShow opens much faster than MSMF on Windows
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        if not Path(src).exists():
            sys.exit(f"Video not found: {src}")
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {src}")
    return cap


def source_name(src: str) -> str:
    return "webcam" if src.isdigit() else Path(src).stem


def resolve_model(name: str) -> str:
    """Keep weights in models/ instead of wherever the script is launched from."""
    p = Path(name)
    if p.exists() or p.parent != Path("."):
        return str(p)
    MODELS.mkdir(exist_ok=True)
    local = MODELS / name
    if not local.exists():  # Ultralytics downloads into the CWD; fetch once, then move
        from ultralytics import YOLO
        YOLO(name)
        if Path(name).exists():
            Path(name).replace(local)
    return str(local) if local.exists() else name
