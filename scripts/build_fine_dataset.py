# -*- coding: utf-8 -*-
"""Build the fine-grained 22-class YOLO dataset from TrashCAN (native classes).

This is the dataset behind the `fine22` model (see train_fine.py). Unlike
build_unified_dataset.py which merges everything into trash/bio/rov, this
script keeps TrashCAN's original 22 categories verbatim.

Protocol notes:
- TrashCAN provides train/val COCO json only - no test split. We keep that.
- Class ids are assigned in the fixed order below (matches fine/data.yaml).
- Same bbox sanity filters as the unified build.

Usage:  python scripts/build_fine_dataset.py
Input:  raw/dataset/instance_version/{train,val}/ + instances_*.json
Output: fine/{images,labels}/{train,val}/ + fine/data.yaml
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "dataset" / "instance_version"
OUT = ROOT / "fine"

# fixed class order (id = position); TrashCAN category names map 1:1
CLASSES = [
    "rov", "plant", "animal_fish", "animal_starfish", "animal_shells",
    "animal_crab", "animal_eel", "animal_etc", "trash_clothing", "trash_pipe",
    "trash_bottle", "trash_bag", "trash_snack_wrapper", "trash_can",
    "trash_cup", "trash_container", "trash_unknown_instance", "trash_branch",
    "trash_wreckage", "trash_tarp", "trash_rope", "trash_net",
]
NAME2ID = {n: i for i, n in enumerate(CLASSES)}


def coco_to_yolo(img_w, img_h, x, y, w, h):
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def main():
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in ("train", "val"):
        data = json.loads((RAW / f"instances_{split}_trashcan.json").read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data["categories"]}
        unknown = set(cats.values()) - set(CLASSES)
        if unknown:
            raise SystemExit(f"unexpected TrashCAN categories {unknown} - check dataset version")
        anns_by_img = {}
        for a in data["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        n_img, boxes = 0, Counter()
        for im in data["images"]:
            img_path = RAW / split / im["file_name"]
            if not img_path.exists():
                continue
            W, H = im["width"], im["height"]
            lines = []
            for a in anns_by_img.get(im["id"], []):
                x, y, w, h = a["bbox"]
                if w <= 1 or h <= 1 or x < 0 or y < 0:
                    continue
                cx, cy, nw, nh = coco_to_yolo(W, H, x, y, w, h)
                if not (0 < nw <= 1 and 0 < nh <= 1):
                    continue
                cid = NAME2ID[cats[a["category_id"]]]
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                boxes[cid] += 1
            dst_img = OUT / "images" / split / img_path.name
            dst_lbl = OUT / "labels" / split / (img_path.stem + ".txt")
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("".join(lines), encoding="ascii")
            n_img += 1
        stats[split] = (n_img, boxes)
        print(f"{split}: {n_img} images")

    (OUT / "data.yaml").write_text(
        "train: images/train\nval: images/val\n\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)),
        encoding="utf-8")
    print("wrote fine/data.yaml")


if __name__ == "__main__":
    main()
