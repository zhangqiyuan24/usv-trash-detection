# -*- coding: utf-8 -*-
"""Export every trained best.pt to ONNX (for onnxruntime / TensorRT benchmarks).

Usage: python export_onnx.py
"""
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

for run in sorted((ROOT / "runs").glob("*_trash")):
    best = run / "weights" / "best.pt"
    if not best.exists():
        continue
    model = YOLO(str(best))
    path = model.export(format="onnx", imgsz=640, simplify=True, dynamic=False, opset=13)
    print(f"exported {run.name}: {path}")
