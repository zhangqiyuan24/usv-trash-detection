# -*- coding: utf-8 -*-
"""Build unified YOLO dataset from Trash-ICRA19 + TrashCan.

Protocol notes:
- Keep each dataset's native train/val/test splits. Both datasets are built from
  consecutive video frames (e.g. bio0000_frame0000001/2/4...), so a random
  re-split would leak near-identical adjacent frames across splits and inflate
  metrics. Native splits respect video groupings.
- Unified 3-class scheme (id, name): 0=trash, 1=bio, 2=rov
    ICRA19  (darknet txt, names plastic/bio/rov):  plastic->0(trash), bio->1, rov->2
    TrashCan (COCO json, 22 classes): animal_* + plant -> 1(bio); rov -> 2; trash_* -> 0(trash)
- Output: <root>/unified/{images,labels}/{train,val,test}/  + data.yaml + stats
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent          # repro/
RAW = ROOT / "raw"
OUT = ROOT / "unified"

BIO_PREFIXES = ("animal_", "plant")                     # TrashCan -> bio(1)

def coco_to_yolo(img_w, img_h, x, y, w, h):
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h

def copy_pair(src_img, label_lines, split, prefix):
    """Copy image into OUT/images/<split>/ and write label txt into OUT/labels/<split>/."""
    stem = prefix + src_img.stem
    dst_img = OUT / "images" / split / (stem + src_img.suffix.lower())
    dst_lbl = OUT / "labels" / split / (stem + ".txt")
    shutil.copy2(src_img, dst_img)
    dst_lbl.write_text("".join(label_lines), encoding="ascii")
    return len(label_lines)

def build_icra19(stats):
    base = RAW / "trash_ICRA19" / "dataset"
    for split in ("train", "val", "test"):
        files = sorted(p for p in (base / split).glob("*.jpg"))
        for img in files:
            lbl = img.with_suffix(".txt")
            # remap: old ids plastic=0 bio=1 rov=2 -> same ids, names only differ
            lines = []
            if lbl.exists():
                for ln in lbl.read_text().splitlines():
                    parts = ln.split()
                    if len(parts) == 5:
                        lines.append(f"{int(parts[0])} {float(parts[1]):.6f} {float(parts[2]):.6f} {float(parts[3]):.6f} {float(parts[4]):.6f}\n")
            n = copy_pair(img, lines, split, "icra19_")
            stats["imgs"][split] += 1
            stats["boxes"][split].update([int(l.split()[0]) for l in lines])
    return stats

def build_trashcan(stats):
    ver = RAW / "dataset" / "instance_version"
    for coco_split, out_split in (("train", "train"), ("val", "val")):
        data = json.loads((ver / f"instances_{coco_split}_trashcan.json").read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data["categories"]}
        def map_class(name):
            if name == "rov":
                return 2
            if name.startswith(BIO_PREFIXES):
                return 1
            return 0
        cat2uni = {cid: map_class(cats[cid]) for cid in cats}
        anns_by_img = {}
        for a in data["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        for im in data["images"]:
            img_path = ver / coco_split / im["file_name"]
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
                lines.append(f"{cat2uni[a['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            copy_pair(img_path, lines, out_split, "trashcan_")
            stats["imgs"][out_split] += 1
            stats["boxes"][out_split].update([int(l.split()[0]) for l in lines])
    return stats

def main():
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    stats = {"imgs": Counter(), "boxes": {s: Counter() for s in ("train", "val", "test")}}
    print("processing Trash-ICRA19 ...")
    build_icra19(stats)
    print("processing TrashCan ...")
    build_trashcan(stats)
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: trash\n  1: bio\n  2: rov\n", encoding="utf-8")
    print("\n=== stats ===")
    total = 0
    for s in ("train", "val", "test"):
        n = stats["imgs"][s]
        total += n
        b = stats["boxes"][s]
        print(f"{s:5s}: {n:6d} images | boxes trash={b[0]} bio={b[1]} rov={b[2]}")
    print(f"total : {total} images")
    (ROOT / "unified_STATS.md").write_text(
        "# Unified dataset stats\n\n"
        "| split | images | trash | bio | rov |\n|---|---|---|---|---|\n"
        + "\n".join(f"| {s} | {stats['imgs'][s]} | {stats['boxes'][s][0]} | {stats['boxes'][s][1]} | {stats['boxes'][s][2]} |"
                    for s in ("train", "val", "test"))
        + f"\n| total | {total} | | | |\n", encoding="utf-8")

if __name__ == "__main__":
    main()
