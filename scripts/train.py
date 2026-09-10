# -*- coding: utf-8 -*-
"""Train one nano-tier YOLO on the unified trash dataset.

All three architectures train under the SAME ultralytics trainer for a clean
architecture-level comparison (v5n uses the yolov5nu re-release).

Usage:  python train.py --model v5n [--epochs 100] [--batch 16]
Re-running the same model auto-resumes from last.pt.
"""
import argparse

from ultralytics import YOLO

from repro_common import ROOT, data_yaml

MODELS = {"v5n": "yolov5nu.pt", "v8n": "yolov8n.pt", "v11n": "yolo11n.pt"}


def weight_path(name):
    """Prefer the local sha256-verified copy in repro/weights/ (GitHub CDN is
    blocked on CN networks; ultralytics would try to auto-download and fail)."""
    local = ROOT / "weights" / name
    return str(local) if local.exists() else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    run_name = f"{args.model}_trash"
    last = ROOT / "runs" / run_name / "weights" / "last.pt"
    if last.exists():
        print(f"[resume] {last}")
        model = YOLO(str(last))
        model.train(resume=True)
        return

    model = YOLO(weight_path(MODELS[args.model]))
    model.train(
        data=data_yaml(),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs"),
        name=run_name,
        exist_ok=True,
        seed=42,
        patience=30,
    )


if __name__ == "__main__":
    main()
