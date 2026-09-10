# USV Trash Detection — English Quick Start

> This is a condensed English overview. The full documentation (setup, dataset protocol, training, evaluation, deployment) is in Chinese: [README.md](README.md) → `docs/1-setup.md` … `docs/6-story-2022.md`.

## What this is

An open reproduction of the detection pipeline from a 2022 robotics-competition project — an autonomous surface vessel for water-surface garbage collection (1st prize, 15th International Advanced Robotics & Simulation Competition, China Simulation Society).

**Honest scope note**: the original 2022 project trained on a *private, self-collected* water-surface dataset, which is not published here. This repo rebuilds the same pipeline end-to-end on two **public** underwater trash datasets — [Trash-ICRA19](https://conservancy.umn.edu/handle/11299/214600) (deep-sea, darknet txt) and [TrashCAN](https://github.com/nathanvasquez/Trash-CAN) (near-shore, COCO, 22 classes) — with no private resources involved. Original project report: [tech-report-2022.pdf](docs/tech-report-2022.pdf) (Chinese).

## Pipeline

| Stage | Script | Notes |
|---|---|---|
| Environment | `scripts/setup_env.ps1` | venv + PyTorch cu124 + ultralytics (CN mirrors, works everywhere) |
| Data download | `scripts/download_datasets.ps1` | ~1.5 GB from official sources, 8-way parallel, resumable |
| Dataset build | `scripts/build_unified_dataset.py` | 3-class unified set (trash / bio / rov); native splits preserved to avoid adjacent-video-frame leakage |
| Smoke test | `scripts/smoke_test.py` | 1-epoch CPU sanity run, ~2 min |
| Train | `scripts/train.py --model {v5n,v8n,v11n}` | identical protocol: 100 ep, batch 16, 640², seed 42, auto-resume |
| Evaluate | `scripts/eval_test.py` | P / R / mAP@0.5 / mAP@0.5:0.95 on the test split |
| Latency | `scripts/bench_infer.py` | FPS on synthetic 640² frames |
| Export | `scripts/export_onnx.py` | opset 13, for onnxruntime / TensorRT (Jetson) |
| Live demo | `scripts/realtime_detect.py` | USB camera real-time detection |

A 22-class fine-grained variant (`build_fine_dataset.py` / `train_fine.py`) is also included.

## Quick start (Windows; Linux works the same)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1        # 1. env
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1 # 2. data (~1.5 GB)
# unzip into raw\ (see docs/2-dataset.md for layout), then:
python scripts\build_unified_dataset.py
python scripts\smoke_test.py
python scripts\train.py --model v5n                                   # 3. train (6–10 h on a 4 GB GPU)
```

**Skip training** — grab Release weights and run the camera demo:

```powershell
curl -L -o runs\v5n_trash\weights\best.pt https://github.com/zhangqiyuan24/usv-trash-detection/releases/download/v1.0/v5n_trash.pt
python scripts\realtime_detect.py --model v5n
```

## Benchmarks

Test split (1,144 imgs / 1,668 instances, ICRA19 test only) — RTX 3050 Ti Laptop, same protocol for all models:

| Model | Params | P | R | mAP@0.5 | mAP@0.5:0.95 | FPS (640²) |
|---|---|---|---|---|---|---|
| YOLOv5n | 2.5M | 0.952 | 0.948 | 0.982 | 0.792 | 63.9 |
| YOLOv8n | 3.0M | 0.948 | 0.961 | 0.985 | 0.801 | 77.9 |
| YOLO11n | 2.6M | 0.950 | 0.961 | 0.984 | 0.805 | 60.9 |
| fine22 (v5n, 22 cls) | 2.5M | — | — | ≈0.66 (val) | — | 66.1 |

⚠️ Test mAP is *not* comparable with training-time val (~0.66–0.70): the test split contains only ICRA19 images (larger, clearer targets) while val mixes in harder TrashCAN frames. The two source datasets have no common external test set. Details: [docs/4-eval-bench.md](docs/4-eval-bench.md).

Historical reference from the 2022 project: YOLOv5n on Jetson Nano 4GB ran 6/11 FPS (FP32, 5W/10W) → 10/18 FPS after INT8 quantization.

## Data license

No dataset images are distributed in this repo — only official-channel download scripts and annotation-conversion scripts. Both upstream datasets are **research-use**; cite the original authors (Watanabe et al. for Trash-ICRA19; Dana et al. for TrashCAN / JAMSTEC) in your work.

## License

Code: [MIT](LICENSE). Dataset images and derived labels remain under their upstream licenses.
