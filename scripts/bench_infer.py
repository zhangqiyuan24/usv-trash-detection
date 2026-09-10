# -*- coding: utf-8 -*-
"""Inference latency benchmark: synthetic 640x640 frames, GPU, after warmup.

Usage:  python scripts/bench_infer.py            # all runs found under runs/
        python scripts/bench_infer.py --device cpu
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    ap.add_argument("--frames", type=int, default=50, help="timed frames per model")
    args = ap.parse_args()

    runs = sorted((ROOT / "runs").glob("*_trash")) + sorted((ROOT / "runs").glob("*_fine*"))
    if not runs:
        print("no trained runs found under runs/ - train first (docs/3-train.md)")
        return

    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for run in runs:
        best = run / "weights" / "best.pt"
        if not best.exists():
            continue
        m = YOLO(str(best))
        for _ in range(5):  # warmup: first calls include CUDA init
            m.predict(img, imgsz=640, device=args.device, verbose=False)
        t0 = time.perf_counter()
        for _ in range(args.frames):
            m.predict(img, imgsz=640, device=args.device, verbose=False)
        dt = (time.perf_counter() - t0) / args.frames * 1000
        print(f"{run.name}: {dt:.1f} ms/frame  {1000/dt:.1f} FPS", flush=True)
    print("BENCH DONE")


if __name__ == "__main__":
    main()
