# -*- coding: utf-8 -*-
"""Shared helpers for the repro scripts (imported by train/eval/smoke).

Runs as plain scripts, so sys.path[0] = this directory and the import works
regardless of cwd.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # .../repro

NAMES = ("trash", "bio", "rov")


def data_yaml(train="images/train", val="images/val", test="images/test"):
    """Write unified/data.yaml with the dataset root resolved on THIS machine,
    return its path.

    Why a runtime-generated yaml instead of a dict or a static file:
    - ultralytics 8.4.x str()-ifies a dict `data` arg (dataset = str(dataset)
      in check_det_dataset), so it MUST be a yaml file path;
    - absolute paths baked on one machine (E:/...) break on another (D:...);
    - CJK path must be utf-8 - the encoding ultralytics' YAML.load uses.
    """
    u = ROOT / "unified"
    # no `path:` key on purpose - ultralytics then resolves relative entries
    # against the yaml's own directory (data/utils.py: "path" falls back to
    # yaml_file.parent), making the file portable across drives/machines
    lines = []
    for k, v in (("train", train), ("val", val), ("test", test)):
        p = Path(v)
        v = p.as_posix() if p.is_absolute() else v
        lines.append(f"{k}: {v}")
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    y = u / "data.yaml"
    y.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(y)
