"""Image, UI, and audit-artifact helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """Return a frame scaled down to ``max_width``, preserving aspect ratio."""
    if frame is None or frame.size == 0:
        raise ValueError("Cannot resize an empty frame.")
    if max_width < 1:
        raise ValueError("max_width must be positive.")
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    return cv2.resize(frame, (max_width, int(height * max_width / width)), interpolation=cv2.INTER_AREA)


def draw_banner(frame: np.ndarray, text: str, color: tuple[int, int, int] = (30, 30, 30)) -> None:
    """Draw a high-contrast banner at the top of a BGR image in-place."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), color, -1)
    cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float, counts: dict[str, int]) -> None:
    """Overlay processing rate and compact detection counts in-place."""
    summary = " | ".join(f"{key}: {value}" for key, value in counts.items() if value)
    draw_banner(frame, f"FPS: {fps:.1f}" + (f"   {summary}" if summary else ""))


def save_snapshot(frame: np.ndarray, directory: str | Path = "logs") -> str:
    """Persist a JPEG snapshot and return its path string."""
    if frame is None or frame.size == 0:
        raise ValueError("Cannot save an empty frame.")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    name = f"violation_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}_{uuid4().hex[:8]}.jpg"
    path = target / name
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"OpenCV could not write snapshot: {path}")
    return str(path)


def normalize_class_names(names: Sequence[str] | dict[int, str]) -> dict[int, str]:
    """Normalize class-name metadata to an integer-keyed mapping."""
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(name) for index, name in enumerate(names)}
