# -*- coding: utf-8 -*-
"""Real-time trash detection from a UVC webcam (Insta360 Ace Pro 2, Webcam mode).

Keys: q = quit, s = save snapshot (CJK-safe via imencode + write_bytes).
Examples:
  python realtime_detect.py                  # fine22 (22 fine classes)
  python realtime_detect.py --model v8n      # 3-class unified
  python realtime_detect.py --cam 1          # pick camera index (see probe_cam.py)
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "fine22": ROOT / "runs" / "v5n_fine22" / "weights" / "best.pt",
    "v5n": ROOT / "runs" / "v5n_trash" / "weights" / "best.pt",
    "v8n": ROOT / "runs" / "v8n_trash" / "weights" / "best.pt",
    "v11n": ROOT / "runs" / "v11n_trash" / "weights" / "best.pt",
}
SNAP_DIR = ROOT / "realtime_snaps"


def open_cam(index):
    """Try DirectShow first (fast open on Windows), fall back to MSMF."""
    for api, tag in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")):
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            return cap, tag
        cap.release()
    return None, "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fine22", choices=sorted(MODELS))
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--only-trash", action="store_true",
                    help="draw only identifiable trash_* classes "
                         "(drop rov/bio/trash_unknown noise)")
    args = ap.parse_args()

    weight = MODELS[args.model]
    if not weight.exists():
        print(f"[ERR] weight not found: {weight}")
        return 2

    cap, api = open_cam(args.cam)
    if cap is None:
        print("[ERR] no camera at this index - run probe_cam.py to list cameras")
        return 3
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[cam] index={args.cam} api={api} "
          f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}@"
          f"{cap.get(cv2.CAP_PROP_FPS):.0f}fps")

    model = YOLO(str(weight))
    cls_filter = None
    if args.only_trash:
        cls_filter = [i for i, n in model.names.items()
                      if n.startswith("trash_")
                      and n != "trash_unknown_instance"]
        keep = ", ".join(model.names[i] for i in sorted(cls_filter))
        print(f"[filter] {len(cls_filter)} classes: {keep}")
    # warmup: build CUDA context before the timing loop (first call is slow)
    model.predict(np.zeros((1080, 1920, 3), np.uint8), imgsz=640,
                  device="0", verbose=False)

    snaps, fps_ema = 0, 0.0
    t_prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] frame grab failed, exiting")
            break
        t0 = time.perf_counter()
        res = model.predict(frame, imgsz=640, conf=args.conf, device="0",
                            classes=cls_filter, verbose=False)[0]
        infer_ms = (time.perf_counter() - t0) * 1000
        ann = res.plot()

        now = time.perf_counter()
        e2e = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        fps_ema = e2e if fps_ema == 0 else 0.9 * fps_ema + 0.1 * e2e
        cv2.putText(ann,
                    f"{args.model}  infer {infer_ms:5.1f} ms  e2e {fps_ema:5.1f} FPS  [q]quit [s]snap",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("realtime detect", ann)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            SNAP_DIR.mkdir(exist_ok=True)
            ok2, buf = cv2.imencode(".jpg", ann)
            if ok2:
                snaps += 1
                p = SNAP_DIR / f"snap_{time.strftime('%H%M%S')}_{snaps}.jpg"
                p.write_bytes(buf.tobytes())
                print(f"[snap] {p}")

    cap.release()
    cv2.destroyAllWindows()
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
