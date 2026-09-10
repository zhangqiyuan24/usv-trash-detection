# -*- coding: utf-8 -*-
"""End-to-end pipeline smoke test: 1 epoch on a 120-image subset.

Catches dataset-format bugs (labels, class ids, paths, yaml, encodings) before
a real run. Works on CPU - safe to execute on any machine with the venv ready.
"""
import sys
from pathlib import Path

from ultralytics import YOLO

from repro_common import ROOT, data_yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
IMG = ROOT / "unified" / "images"


def make_list(split, n, out):
    files = sorted((IMG / split).glob("*.jpg"))[:n]
    # utf-8: ultralytics 8.4.x opens txt lists with encoding="utf-8" (base.py),
    # and paths contain CJK on this machine
    out.write_text("\n".join(str(f) for f in files), encoding="utf-8")
    return len(files)


def main():
    n_tr = make_list("train", 120, ROOT / "unified" / "train_mini.txt")
    n_va = make_list("val", 40, ROOT / "unified" / "val_mini.txt")
    print(f"mini set: train={n_tr} val={n_va}")
    # absolute txt entries (a txt list exercises a different loader branch
    # than the directory entries the real runs use)
    data = data_yaml(
        train=str(ROOT / "unified" / "train_mini.txt"),
        val=str(ROOT / "unified" / "val_mini.txt"),
    )
    model = YOLO(str(ROOT / "weights" / "yolov8n.pt"))
    model.train(
        data=data, epochs=1, imgsz=320, batch=8, device="cpu", workers=0,
        project=str(ROOT / "runs"), name="smoke", exist_ok=True, val=True,
        plots=False,
    )
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
