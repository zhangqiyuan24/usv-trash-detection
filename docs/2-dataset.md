# 2. 数据集构建

## 上游数据集（先读许可）

| 数据集 | 内容 | 规模 | 获取 |
|---|---|---|---|
| Trash-ICRA19 | 深海视频分帧（JAMSTEC J-EDI，AUV 拍摄） | 7,212 张，darknet txt 标注 | [UMN Conservancy](https://conservancy.umn.edu/handle/11299/214600) |
| TrashCAN | 近岸水下（instance version） | 5,000+ 张，COCO json，22 类 | [GitHub](https://github.com/nathanvasquez/Trash-CAN) |

**两者均为研究用途。本仓库不分发任何图像**，只提供官方渠道下载脚本与标注转换脚本。在你的论文/项目里请引用上游。

## 下载

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
# 可选：-OutDir D:\bigdisk\downloads 指定下载目录
```

- 并行 8 线程分块下载 + 断点续传（中断后**重跑同一命令**即可续传）
- 产物：`downloads\trash_ICRA19.zip`（~980 MB）、`downloads\trashcan_dataset.zip`（~527 MB）
- 下载慢是 Conservancy 服务端限速，属正常；两文件合计约 15-40 分钟

## 解压布局

解压到仓库根目录的 `raw\`（名字不能错，构建脚本按此寻路）：

```
usv-trash-detection/
└── raw/
    ├── trash_ICRA19/
    │   └── dataset/
    │       ├── train/   (*.jpg + 同名 .txt)
    │       ├── val/
    │       └── test/
    └── dataset/                      # TrashCAN 解压后自带这层目录名
        └── instance_version/
            ├── train/  val/
            ├── instances_train_trashcan.json
            └── instances_val_trashcan.json
```

## 构建

```powershell
# 三类统一数据集（trash/bio/rov，v5n/v8n/v11n 用）
python scripts\build_unified_dataset.py

# 22 类细粒度数据集（fine22 用，可选）
python scripts\build_fine_dataset.py
```

产物：`unified\`（或 `fine\`）下的 `images\{train,val,test}` + `labels\...` + `data.yaml`，并打印 split/类别统计。

## 划分协议（为什么不是随机重划分）

两个数据集都来自**连续视频分帧**（同一目标相邻帧几乎相同）。随机重划分会让近似重复的帧横跨 train/val/test，指标虚高。因此构建脚本**保留各自原生划分**不做重排——TrashCAN 没有 test 划分，所以 unified 的 test split 只含 ICRA19 test 图像（这也是 [4-eval-bench.md](4-eval-bench.md) 里 test 与 val 指标不可直接比较的原因）。

## 类别映射

| 上游类别 | unified 类 |
|---|---|
| ICRA19: plastic | 0 = trash |
| ICRA19/TrashCAN: bio、animal_*、plant | 1 = bio |
| rov | 2 = rov |
| TrashCAN: trash_*（其余全部） | 0 = trash |

fine 数据集不做合并，保留 TrashCAN 原生 22 类。

## 验证

```powershell
python scripts\smoke_test.py     # 120 张小子集 CPU 跑 1 epoch，2 分钟内应打印 SMOKE TEST PASSED
```

冒烟测试能抓住标签格式、类别 id 越界、路径、编码等几乎所有数据问题。**先跑它再开训练**。

之后进入 [3-train.md](3-train.md)。
