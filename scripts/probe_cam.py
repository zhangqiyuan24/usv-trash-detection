# -*- coding: utf-8 -*-
"""List UVC cameras: which indices open, at what resolution. No GUI."""
import cv2

for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ok, frame = cap.read()
        h, w = frame.shape[:2] if ok and frame is not None else (0, 0)
        print(f"index {i}: OPEN {w}x{h}")
        cap.release()
    else:
        print(f"index {i}: -")
