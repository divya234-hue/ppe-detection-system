"""Export trained Ultralytics YOLO PPE weights to a portable ONNX graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    """Parse CLI arguments and export the requested YOLO model."""
    parser = argparse.ArgumentParser(description="Export YOLO PPE .pt weights to optimized ONNX.")
    parser.add_argument("weights", type=Path, help="Path to a trained .pt file")
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference/export size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX operator-set version")
    parser.add_argument("--dynamic", action="store_true", help="Allow variable input dimensions")
    parser.add_argument("--simplify", action="store_true", help="Simplify the graph when onnxslim is installed")
    args = parser.parse_args()
    if args.weights.suffix.lower() != ".pt" or not args.weights.is_file():
        parser.error("weights must point to an existing .pt model")
    try:
        output = YOLO(str(args.weights)).export(format="onnx", imgsz=args.imgsz, opset=args.opset, dynamic=args.dynamic, simplify=args.simplify)
    except Exception as error:
        raise SystemExit(f"ONNX export failed: {error}") from error
    print(f"ONNX model exported: {output}")


if __name__ == "__main__":
    main()
