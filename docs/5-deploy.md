# 5. 部署与实时演示

## ONNX 导出

```powershell
python scripts\export_onnx.py
# 产物：runs\<模型>\weights\best.onnx（opset 13，静态 640，已 simplify）
```

导出后可以脱离 ultralytics/PyTorch 推理：

```python
# pip install onnxruntime(-gpu)
import cv2, numpy as np, onnxruntime as ort
sess = ort.InferenceSession(r"runs\v5n_trash\weights\best.onnx",
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
img = cv2.resize(cv2.imread("test.jpg"), (640, 640))
blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
out = sess.run(None, {"images": blob})[0]   # (1, N, 6): x,y,w,h,conf,cls（按导出版本可能有差异）
```

## Jetson（Nano / Orin）部署路线

2022 年参赛项目在 Jetson Nano 4GB 上的实测经验（详见[技术报告](tech-report-2022.pdf)第六节）：

| 形态 | 5W 模式 | 10W 模式 |
|---|---|---|
| YOLOv5n FP32 | 6 FPS | 11 FPS |
| YOLOv5n INT8 量化 | 10 FPS | **18 FPS** |

推荐路线（以 Orin/Xavier 系列为例，Nano 已停产但流程相同）：

1. 刷 JetPack（自带 TensorRT）
2. ONNX → TensorRT engine：
   ```bash
   trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
   # INT8 需标定：--int8 --calib=calib.cache（或用 ultralytics 的 format='engine' 导出）
   ```
3. `nvpmodel -m 0` 解锁最大功率档再测帧率
4. 摄像头用 `realtime_detect.py` 同款 OpenCV 采集逻辑即可

显存参考：640 输入的 nano 级模型 INT8 engine 约 6-10 MB，Nano 4GB 无压力；瓶颈在算力不在内存。

## 摄像头实时演示（PC）

```powershell
python scripts\probe_cam.py                  # 先看有哪些摄像头、哪个索引可用
python scripts\realtime_detect.py --model v5n --cam 0
```

- 按键：`q` 退出，`s` 存快照到 `realtime_snaps\`
- 默认模型 `fine22`（22 类，识别粒度细）；`--model v5n/v8n/v11n` 为三类
- Windows 下自动优先 DirectShow 打开（比 MSMF 快），失败自动回退
- 用 Release 权重即可演示，无需训练

## 演示效果提升小技巧

- 俯拍、距离 1-3 m、光照均匀时效果最好（与上游数据的拍摄条件一致）
- 逆光/浑水/夜间是已知弱项（域外），改用 fine22 时 22 类粒度对浑浊目标更稳
- 想压误检可调置信度阈值：`realtime_detect.py` 里 `conf=` 参数（默认 0.25；提到 0.4 换取干净画面）
