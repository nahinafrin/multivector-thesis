# sample_resources.ps1 -- 1 Hz resource sampler for the per-query overhead measurement.
# Writes CSV rows: timestamp, GPU memory used (MiB), GPU util (%), resident memory (MiB)
# of all python.exe and all ollama*.exe processes, and system available memory (MiB).
# Stop it by creating the file named in -StopFile.
param(
    [string]$Out = "resources.csv",
    [string]$StopFile = "stop.flag"
)
"t_unix,gpu_mem_mib,gpu_util_pct,python_rss_mib,ollama_rss_mib,sys_avail_mib" | Out-File -Encoding ascii $Out
while (-not (Test-Path $StopFile)) {
    $t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    $g = (& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) -split ","
    $py = (Get-Process -Name python -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum
    $ol = (Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum
    $av = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory  # KiB
    $line = [string]::Format([Globalization.CultureInfo]::InvariantCulture,
        "{0:F1},{1},{2},{3:F0},{4:F0},{5:F0}", $t, $g[0].Trim(), $g[1].Trim(), ($py/1MB), ($ol/1MB), ($av/1KB))
    $line | Out-File -Append -Encoding ascii $Out
    Start-Sleep -Milliseconds 1000
}
