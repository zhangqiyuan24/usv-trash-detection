# -*- coding: utf-8 -*-
"""List fine22 class names + instance histogram from train labels."""
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
names = {}
for line in (ROOT / "fine" / "data.yaml").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and line[0].isdigit() and ":" in line:
        i, n = line.split(":", 1)
        names[int(i)] = n.strip()

hist = Counter()
for f in (ROOT / "fine" / "labels" / "train").glob("*.txt"):
    for ln in f.read_text().splitlines():
        if ln.strip():
            hist[int(ln.split()[0])] += 1

total = sum(hist.values())
print(f"total train instances: {total}")
for i in sorted(names):
    print(f"{i:2d} {names[i]:28s} {hist.get(i,0):5d}")
