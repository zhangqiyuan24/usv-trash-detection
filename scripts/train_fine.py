# -*- coding: utf-8 -*-
"""Train a nano-tier YOLO on the fine-grained 22-class TrashCan dataset.

Standalone on purpose: tonight's 3-model chain imports train.py / repro_common
while it runs - this file touches neither, so deploying it mid-chain is safe.

Usage:  python train_fine.py --model v5n [--epochs 100] [--batch 16]
Re-running the same model auto-resumes from last.pt.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"v5n": "yolov5nu.pt", "v8n": "yolov8n.pt", "v11n": "yolo11n.pt"}

def weight_path(name):
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

    run_name = f"{args.model}_fine22"
    last = ROOT / "runs" / run_name / "weights" / "last.pt"
    if last.exists():
        print(f"[resume] {last}")
        model = YOLO(str(last))
        model.train(resume=True)
        return

    model = YOLO(weight_path(MODELS[args.model]))
    model.train(
        data=str(ROOT / "fine" / "data.yaml"),
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
