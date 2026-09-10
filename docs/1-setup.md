# 1. 环境搭建

目标：一台有 NVIDIA 显卡（≥4GB 显存）的 Windows 或 Linux 机器。纯 CPU 也能跑通数据构建、冒烟测试和实时演示（慢），只有训练建议用 GPU。

## Windows（推荐路径）

前提：已装 **Python 3.10+**（[python.org](https://www.python.org/downloads/) 下载，安装时勾选 *Add python.exe to PATH*）。

```powershell
cd usv-trash-detection
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

脚本会：

1. 在项目目录下创建 `venv\`
2. pip 缓存与临时文件重定向到项目内（避免撑爆 C 盘）
3. 从清华镜像装 pip/ultralytics（国内网络友好）
4. 从 PyTorch 官方源装 `torch + torchvision`（CUDA 12.4 轮子，约 2.5 GB）
5. 跑一次 CUDA 自检并打印 `ENV READY`

最后一行看到 `ENV READY` 且 `cuda True` 即成功。

## Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # 或 cu121/cu118 按驱动
pip install ultralytics opencv-python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

无需国内镜像的话，把镜像参数去掉即可。

## 常见问题

| 现象 | 处理 |
|---|---|
| torch 下载中断 | 重跑一遍 `setup_env.ps1`，pip 会从缓存续传 |
| `cuda False` | 驱动太旧：更新 NVIDIA 驱动；或改用 CPU 轮子（`--index-url .../whl/cpu`）先跑通流程 |
| 训练报 CUDA out of memory | `python scripts\train.py --model v5n --batch 8`（4GB 卡的保险档） |
| PowerShell 执行策略限制 | 用示例中的 `-ExecutionPolicy Bypass` 参数，不需要改系统全局策略 |
| 公司/校园网拦截 pytorch.org | 用手机热点或配置代理后重试 |

## 之后

进入 [2-dataset.md](2-dataset.md) 准备数据。
