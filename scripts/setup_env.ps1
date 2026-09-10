# Setup Python env for training on the 3050 Ti laptop (ASCII only, PS 5.1 safe)
# Usage: powershell -ExecutionPolicy Bypass -File setup_env.ps1
$root = Split-Path $PSScriptRoot -Parent
$py   = "$root\venv\Scripts\python.exe"
$pip  = "$root\venv\Scripts\pip.exe"

# Keep pip cache and installer temp files OFF the C: drive (small C: partition).
# Everything lands inside the project folder on D: instead.
$cache = "$root\pip_cache"
$tmp   = "$root\tmp"
New-Item -ItemType Directory -Force $cache | Out-Null
New-Item -ItemType Directory -Force $tmp | Out-Null
$env:PIP_CACHE_DIR = $cache
$env:TMP = $tmp
$env:TEMP = $tmp

# report free space so disk pressure is visible early
Get-PSDrive C, D | ForEach-Object { Write-Host ("disk {0}: {1:N1} GB free" -f $_.Name, ($_.Free/1GB)) }

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: python not on PATH. Install Python 3.10+ (check 'Add python.exe to PATH' during setup), then rerun this script."
    exit 1
}
if (-not (Test-Path $py)) {
    python -m venv "$root\venv"
}
& $pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# PyTorch CUDA wheels (Ampere 3050 Ti fully supported). ~2.5 GB, takes a while;
# if it stalls, just rerun - pip resumes from its cache.
# cu124 (torch 2.6.0) is the first line with Python 3.13 wheels; works on the
# 546 driver (CUDA 12.3) via minor-version compatibility (SASS cubins).
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# ultralytics via the tuna mirror (pypi.org direct is unreliable on CN networks)
& $pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
& $py -c "import torch; x = torch.zeros(4, device='cuda'); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'cudatest', x.sum().item())"
Write-Host "ENV READY"
