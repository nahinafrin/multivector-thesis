# pin_environment.ps1
# Run from the repo root. Captures the EXACT environment that produced the
# n=200 external result, so future runs (or a reviewer trying to reproduce
# this work) aren't guessing at versions.

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\sarke\Desktop\Multivector methodology-local api extension"

# 1. Exact pip package versions (not just names -- pip freeze pins versions)
& ".\.venv311\Scripts\python.exe" -m pip freeze | Out-File -Encoding utf8 "step4 dataset\requirements-lock.txt"

# 2. Python + OS + Ollama versions and model digests, all in one file
$info = @()
$info += "Python: " + (& ".\.venv311\Scripts\python.exe" --version)
$info += "Ollama CLI: " + (ollama --version)
$info += ""
$info += "--- ollama list (model digests) ---"
$info += (ollama list | Out-String)
$info += "--- OS ---"
$info += (Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture | Out-String)
$info | Out-File -Encoding utf8 "step4 dataset\environment_info.txt"

Write-Host "Wrote step4 dataset\requirements-lock.txt and environment_info.txt"
