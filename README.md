# USV Trash Detection —— 水域垃圾检测全流程工具链

[English Quick Start](README_EN.md)

2022 年国际先进机器人及仿真技术大赛一等奖项目「无人水面垃圾清理船」算法管线的开源复刻：从公开数据集构建、统一协议训练、测试集评测、延迟基准，到 ONNX 导出与摄像头实时检测，**每一步都可以在本仓库内复现，不依赖任何私有资源**。

> **诚实声明（先读这个）**：当年的参赛项目使用自建水面垃圾数据集（校区水域自采 + 公开水面数据），该数据未随本仓库公开。本仓库用两个**公开**水下垃圾数据集（[Trash-ICRA19](https://conservancy.umn.edu/handle/11299/214600) + [TrashCAN](https://github.com/nathanvasquez/Trash-CAN)）完整复刻了同一条算法管线。当年的算法工作详见[技术报告](docs/tech-report-2022.pdf)。

## 这个仓库能做什么

| 能力 | 说明 |
|---|---|
| 🗂 数据集构建 | 一键下载两个公开数据集（并行分块、断点续传），转换成统一 YOLO 格式 |
| 🎯 三类统一训练 | trash / bio / rov 三类，v5n / v8n / v11n 三个 nano 级模型同协议对比 |
| 🔍 细粒度扩展 | TrashCAN 原生 22 类的 fine-tune（`fine22`） |
| 📊 测试集评测 + 延迟基准 | 测试集 P/R/mAP + GPU 推理延迟（FPS） |
| 📦 ONNX 导出 | 供 onnxruntime / TensorRT（Jetson）使用 |
| 📷 实时检测演示 | USB 摄像头实时推理，键盘交互，快照保存 |

## 快速开始

三步跑通（Windows 为主，Linux 同样适用，见 [docs/1-setup.md](docs/1-setup.md)）：

```powershell
# 1. 环境（Python 3.10+，装 venv + PyTorch cu124 + ultralytics，国内镜像）
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# 2. 数据（下载约 1.5 GB → 解压到 raw\ → 构建统一数据集）
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
#    解压后运行构建 + 冒烟测试（CPU 上 2 分钟，验证数据格式无误）
python scripts\build_unified_dataset.py
python scripts\smoke_test.py

# 3. 训练（一块 4GB 显存的卡即可，如 RTX 3050 Ti；约 6-10 小时/模型）
python scripts\train.py --model v5n
```

**不想训练？** 直接下载预训练权重跑实时检测：

```powershell
# 从 Release 下载权重（约 5-6 MB/个）
curl -L -o runs\v5n_trash\weights\best.pt https://github.com/zhangqiyuan24/usv-trash-detection/releases/download/v1.0/v5n_trash.pt
python scripts\realtime_detect.py --model v5n
```

## 基准结果

**测试集评测**（unified test split，1,144 张 / 1,668 个目标，ICRA19 test；训练 100 epoch，同协议：seed 42 / batch 16 / 640×640 / patience 30）：

| 模型 | 参数量 | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| YOLOv5n | 2.5M | 0.952 | 0.948 | 0.982 | 0.792 |
| YOLOv8n | 3.0M | 0.948 | 0.961 | 0.985 | 0.801 |
| YOLO11n | 2.6M | 0.950 | 0.961 | 0.984 | 0.805 |
| fine22（22 类细粒度，v5n） | 2.5M | — | — | ≈0.66（val） | — |

**推理延迟**（RTX 3050 Ti Laptop 4GB，640×640 合成帧，预热后 50 帧均值）：

| 模型 | ms/帧 | FPS |
|---|---|---|
| YOLOv5n | 15.7 | 63.9 |
| YOLOv8n | 12.8 | 77.9 |
| YOLO11n | 16.4 | 60.9 |
| fine22 | 15.1 | 66.1 |

**历史参照（2022 年 Jetson Nano 实测，来自当年项目）**：FP32 在 5W/10W 功耗档只有 6/11 FPS，INT8 量化后 10/18 FPS——当年的边缘部署经验见[技术报告第六节](docs/tech-report-2022.pdf)。

> ⚠️ **关于指标的说明**：上表测试集 mAP 明显高于训练期 val 指标（~0.66-0.70），原因是**两个数据集没有统一的外部测试集**——unified 的 test split 只含 ICRA19 test 图像（TrashCAN 没有 test 划分），目标更大更清晰；训练期 val 则混合了更难的 TrashCAN 图像。两组数字不能直接比较，复现时请以你自己机器上的完整日志为准。详见 [docs/4-eval-bench.md](docs/4-eval-bench.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [1-setup.md](docs/1-setup.md) | 环境搭建（Windows/Linux，国内镜像，常见安装问题） |
| [2-dataset.md](docs/2-dataset.md) | 数据集下载、目录结构、构建协议、**许可说明** |
| [3-train.md](docs/3-train.md) | 训练协议、显存建议、断点续训、过夜训练 |
| [4-eval-bench.md](docs/4-eval-bench.md) | 评测方法、结果复现、指标口径解释 |
| [5-deploy.md](docs/5-deploy.md) | ONNX 导出、Jetson 部署、实时演示 |
| [6-story-2022.md](docs/6-story-2022.md) | 2022 年参赛项目背景与复刻关系 |
| [tech-report-2022.pdf](docs/tech-report-2022.pdf) | 当年算法技术报告（复盘整理稿） |

## 数据许可（重要）

本仓库**不分发任何数据集图像**，只提供：

- 下载脚本（官方渠道）
- 标注格式转换脚本（我们生成的 YOLO 标签派生自上游标注）

两个上游数据集均为**研究用途**，使用请在你的成果中引用原始出处：

- **Trash-ICRA19**：Watanabe et al., 基于 JAMSTEC J-EDI 深海影像库，[UMN Conservancy 页面](https://conservancy.umn.edu/handle/11299/214600)
- **TrashCAN**：Dana et al. (JAMSTEC)，[GitHub 页面](https://github.com/nathanvasquez/Trash-CAN)

如果你需要把转换后的标签用于研究之外的用途，请先确认上游许可。

## 结构

```
usv-trash-detection/
├── scripts/                  # 全部可执行脚本（无 Claude/IDE 依赖，普通 Python/PowerShell）
│   ├── download_datasets.ps1     # 数据集并行分块下载（断点续传）
│   ├── build_unified_dataset.py  # 构建三类统一数据集（防相邻帧泄漏的划分协议）
│   ├── build_fine_dataset.py     # 构建 TrashCAN 22 类细粒度数据集
│   ├── setup_env.ps1             # 一键环境
│   ├── smoke_test.py             # 冒烟测试（CPU 2 分钟）
│   ├── train.py / train_fine.py  # 统一协议训练（断点续训）
│   ├── eval_test.py / eval_fine.py
│   ├── bench_infer.py            # GPU 延迟基准
│   ├── export_onnx.py            # ONNX 导出
│   ├── realtime_detect.py        # 摄像头实时检测
│   └── probe_cam.py / class_hist.py  # 辅助工具
├── docs/                     # 分步文档 + 当年技术报告
├── raw/                      # （自建）上游数据集解压处，不入库
├── unified/ fine/            # （自建）构建产物，不入库
└── runs/                     # （自建）训练输出，不入库
```

## 已知限制

- 两数据集无统一外部测试集，test 指标与 val 指标不可直接比较（见上文说明）
- 全部为水下/深海口味的数据；水面漂浮视角（当年项目场景）未包含
- 训练协议面向 4GB 显存的入门卡（batch 16 / 100 epoch），大卡可自行调大

## License

代码以 [MIT](LICENSE) 发布；数据集图像不随仓库分发，其许可归上游所有（见上节）。

## 致谢

- 2022 年参赛团队「海丝一队」全体成员
- Trash-ICRA19 与 TrashCAN 的作者与 JAMSTEC
- [ultralytics](https://github.com/ultralytics/ultralytics)
