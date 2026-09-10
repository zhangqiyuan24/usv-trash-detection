# Download the two public source datasets (Trash-ICRA19 + TrashCAN) from UMN
# Conservancy. Parallel chunked downloader with resume (ASCII only - PS 5.1
# reads no-BOM files as ANSI).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
#   powershell ... -OutDir D:\data            # custom download location
#
# Output: <OutDir>\trash_ICRA19.zip (~980 MB), <OutDir>\trashcan_dataset.zip (~527 MB)
# Then unzip both into  raw\  (see docs/2-dataset.md for the expected layout).
param(
    [string]$OutDir = ""
)
$ErrorActionPreference = 'Stop'
if (-not $OutDir) { $OutDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'downloads' }
$UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
$Base = 'https://conservancy.umn.edu/server/api/core/bitstreams'

# Dataset pages (cite these in derived work; research use only):
#   Trash-ICRA19: https://conservancy.umn.edu/handle/11299/214600  (deep-sea, JAMSTEC J-EDI)
#   TrashCAN:     https://github.com/nathanvasquez/Trash-CAN        (harbor, instance version)
$Files = @(
    @{ id = '0239b06a-512e-49c3-80aa-ba33371e11de'; name = 'trash_ICRA19.zip' },    # 980MB
    @{ id = 'fffa78cc-3abf-42b9-b019-bc15a953a8e9'; name = 'trashcan_dataset.zip' } # 527MB
)

function Get-TotalSize([string]$id) {
    $h = curl.exe -s -D - -o NUL -A $UA -H 'Range: bytes=0-0' "$Base/$id/content"
    $line = ($h | Select-String -Pattern 'Content-Range' | Select-Object -First 1).Line
    if ($null -eq $line -or $line -notmatch '/(\d+)\s*$') { throw "cannot get size: $line" }
    return [long]$Matches[1]
}

function Get-Parallel([string]$id, [string]$dest, [long]$chunk = 128MB) {
    $total = Get-TotalSize $id
    Write-Host ("[{0}] total {1:N0} bytes" -f (Split-Path $dest -Leaf), $total)
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -eq $total)) {
        Write-Host "already complete, skip"
        return
    }
    $tmp = "$dest.parts"
    New-Item -ItemType Directory -Force $tmp | Out-Null

    $ranges = @()
    for ($s = 0L; $s -lt $total; $s += $chunk) {
        $e = [long][Math]::Min($s + $chunk, $total) - 1
        $ranges += , @($s, $e)
    }

    # pass 1: parallel, 8 concurrent
    $done = 0
    for ($b = 0; $b -lt $ranges.Count; $b += 8) {
        $last = [Math]::Min($b + 7, $ranges.Count - 1)
        $procs = @()
        for ($i = $b; $i -le $last; $i++) {
            $s, $e = $ranges[$i]
            $part = "$tmp\$i.part"
            if ((Test-Path $part) -and ((Get-Item $part).Length -eq ($e - $s + 1))) { $done++; continue }
            $procs += Start-Process -FilePath curl.exe -ArgumentList @(
                '-s', '-o', $part, '-A', $UA,
                '-H', "Range: bytes=$s-$e", "$Base/$id/content"
            ) -PassThru -WindowStyle Hidden
        }
        if ($procs.Count) { $procs | Wait-Process }
        $done = [Math]::Min($done + $procs.Count, $ranges.Count)
        Write-Host ("  {0}% ({1}/{2} chunks)" -f [int](100 * $done / $ranges.Count), $done, $ranges.Count)
    }

    # pass 2: sequential repair of short/missing parts
    for ($i = 0; $i -lt $ranges.Count; $i++) {
        $s, $e = $ranges[$i]
        $part = "$tmp\$i.part"
        $want = $e - $s + 1
        if ((Test-Path $part) -and ((Get-Item $part).Length -eq $want)) { continue }
        Write-Host "  repairing chunk $i"
        curl.exe -s -o $part -A $UA -H "Range: bytes=$s-$e" "$Base/$id/content"
        if ((Get-Item $part).Length -ne $want) { throw "chunk $i failed twice - rerun this script" }
    }

    # concat + verify
    $fs = [System.IO.File]::OpenWrite($dest)
    $fs.SetLength(0)
    for ($i = 0; $i -lt $ranges.Count; $i++) {
        $bytes = [System.IO.File]::ReadAllBytes("$tmp\$i.part")
        $fs.Write($bytes, 0, $bytes.Length)
    }
    $fs.Close()
    if ((Get-Item $dest).Length -ne $total) { throw "concat size mismatch" }
    Remove-Item -Recurse -Force $tmp
    Write-Host ("[{0}] DONE" -f (Split-Path $dest -Leaf))
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
foreach ($f in $Files) {
    Get-Parallel $f.id (Join-Path $OutDir $f.name)
}
Write-Host "ALL DOWNLOADS COMPLETE"
