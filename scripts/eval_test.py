# -*- coding: utf-8 -*-
"""test split re-eval, take 3: __main__ guard + workers=0 (Windows spawn safe)."""
import sys
from ultralytics import YOLO
from repro_common import ROOT, data_yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    for run in sorted((ROOT / "runs").glob("*_trash")):
        best = run / "weights" / "best.pt"
        if not best.exists():
            continue
        m = YOLO(str(best))
        r = m.val(data=data_yaml(), split="test", imgsz=640, batch=16,
                  device="0", plots=False, workers=0)
        d = r.results_dict
        print(f"{run.name} TEST P={d.get('metrics/precision(B)',0):.4f} "
              f"R={d.get('metrics/recall(B)',0):.4f} "
              f"mAP50={d.get('metrics/mAP50(B)',0):.4f} "
              f"mAP50-95={d.get('metrics/mAP50-95(B)',0):.4f}", flush=True)
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
