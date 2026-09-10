# 3. 训练

## 统一协议

三个模型（v5n / v8n / v11n）刻意使用**完全相同**的训练配置，保证架构间对比公平：

| 项 | 值 |
|---|---|
| epoch | 100 |
| batch | 16（4GB 显存档；≥8GB 可上 32） |
| 输入 | 640 × 640 |
| 预训练权重 | COCO（仓库 `weights\` 里有本地副本，国内网络不需访问 GitHub） |
| seed | 42 |
| patience（早停） | 30 |
| 断点续训 | 自动：`runs\<模型>\weights\last.pt` 存在即 resume |

## 启动

```powershell
# venv 激活后（Windows: venv\Scripts\activate）
python scripts\train.py --model v5n            # v8n / v11n 同理
python scripts\train_fine.py --model v5n       # 22 类细粒度（fine22）
```

输出在 `runs\<模型名>_trash\`（或 `runs\v5n_fine22\`）：`weights\{best,last}.pt`、`results.csv`、混淆矩阵、PR 曲线等图。

## 看进度

```powershell
# 每天看一眼最后三行（epoch, loss, P, R, mAP50, mAP50-95 ...）
Get-Content runs\v5n_trash\results.csv -Tail 3
```

参考时长（RTX 3050 Ti Laptop 4GB，batch 16）：**约 6-10 小时/模型**。

## 过夜训练小抄

- Windows 电源计划设为"不睡眠"：`powercfg /change standby-timeout-ac 0`（以及 `monitor-timeout-ac` 随意，关屏不影响训练）
- 训练命令本身可断点续训：中途断电/重启后**重跑同一条命令**即可从 last.pt 继续
- 三个模型连跑可用一个简单批处理按顺序执行三条 train 命令，无需额外工具

## 显存不够怎么办

1. `--batch 8`（再不行 `--batch 4` + `--imgsz 480`）
2. 关掉其他占显存的程序（浏览器硬件加速、游戏）
3. `nvidia-smi` 确认没有僵尸进程占着内存

## 什么时候算训练完

- epoch 到 100，或 patience=30 触发早停
- `results.csv` 里 mAP50 连续 20+ epoch 不再上升基本就到位了
- 最终以 `weights\best.pt` 为准（验证集最优 epoch 的权重），不要用 last.pt 做对比

之后进入 [4-eval-bench.md](4-eval-bench.md)。
