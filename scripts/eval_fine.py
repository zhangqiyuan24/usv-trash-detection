# -*- coding: utf-8 -*-
"""Evaluate fine22 best.pt on its val split (best-epoch checkpoint, not last)."""
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

m = YOLO(str(ROOT / "runs" / "v5n_fine22" / "weights" / "best.pt"))
r = m.val(data=str(ROOT / "fine" / "data.yaml"), split="val", imgsz=640, batch=16, device="0")
d = r.results_dict
print("fine22 best.pt  P:", round(d.get("metrics/precision(B)", 0), 4),
      "R:", round(d.get("metrics/recall(B)", 0), 4),
      "mAP50:", round(d.get("metrics/mAP50(B)", 0), 4),
      "mAP50-95:", round(d.get("metrics/mAP50-95(B)", 0), 4))
