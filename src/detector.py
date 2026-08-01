"""Model-agnostic, real-time PPE detector supporting Ultralytics PT and ONNX models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .utils import draw_fps, normalize_class_names

PPE_CLASSES = {"hardhat", "hard_hat", "helmet", "safety_vest", "vest", "mask", "face_mask"}
VIOLATION_CLASSES = {"no_hardhat", "no_helmet", "no_vest", "no_mask", "missing_hardhat", "missing_vest", "missing_mask"}


@dataclass(frozen=True)
class Detection:
    """A single model prediction in image coordinates."""

    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    violation: bool


class PPEDetector:
    """Run PPE-trained YOLO PT or ONNX weights and annotate camera frames.

    Train or obtain PPE-specific weights exposing labels such as ``Hardhat``,
    ``Safety Vest``, ``Mask``, and negative labels such as ``no_helmet``.
    """

    def __init__(self, model_path: str | Path, confidence: float = 0.45, device: str | None = None) -> None:
        path = Path(model_path)
        if path.suffix.lower() not in {".pt", ".onnx"}:
            raise ValueError("model_path must end in .pt or .onnx")
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1].")
        self.model_path, self.confidence, self.device = path, confidence, device
        try:
            self.model = YOLO(str(path), task="detect")
        except Exception as error:
            raise RuntimeError(f"Unable to load model '{path}': {error}") from error
        self.names = normalize_class_names(getattr(self.model, "names", {}))

    @staticmethod
    def _canonical(label: str) -> str:
        return label.strip().lower().replace(" ", "_").replace("-", "_")

    def process_frame(self, frame: np.ndarray, target_class: str = "All") -> tuple[np.ndarray, dict[str, int], list[Detection], float]:
        """Infer, annotate, and return ``(image, counts, detections, fps)``."""
        if frame is None or frame.size == 0:
            raise ValueError("process_frame received an empty frame")
        started = time.perf_counter()
        try:
            results = self.model.predict(frame, conf=self.confidence, device=self.device, verbose=False)
        except Exception as error:
            raise RuntimeError(f"Inference failed: {error}") from error
        annotated, detections = frame.copy(), []
        counts = {"compliant": 0, "violations": 0, "people": 0}
        result = results[0]
        names = normalize_class_names(getattr(result, "names", self.names))
        for box in result.boxes or []:
            class_id, score = int(box.cls.item()), float(box.conf.item())
            label = names.get(class_id, f"class_{class_id}")
            if target_class != "All" and self._canonical(label) != self._canonical(target_class):
                continue
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            canonical = self._canonical(label)
            violation = canonical in VIOLATION_CLASSES or canonical.startswith(("no_", "missing_"))
            ppe = canonical in PPE_CLASSES
            color = (0, 0, 255) if violation else ((0, 180, 0) if ppe else (255, 160, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {score:.0%}", (x1, max(52, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            detections.append(Detection(label, score, (x1, y1, x2, y2), violation))
            if violation:
                counts["violations"] += 1
            elif ppe:
                counts["compliant"] += 1
            elif canonical == "person":
                counts["people"] += 1
        fps = 1.0 / max(time.perf_counter() - started, 1e-6)
        draw_fps(annotated, fps, counts)
        return annotated, counts, detections, fps
