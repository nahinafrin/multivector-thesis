# rescale_planted200_arms.ps1
# Re-run full + norefuse ON arms at n=200 (OFF arm already in planted200/off.jsonl).
# Then score ASR and re-pair benign cost (n=200) with the scaled ASR reports.
#
# Prereqs: Ollama on http://localhost:11434, venv active, kb index built.
# Run from `step4 dataset`:
#   .\rescale_planted200_arms.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Slice = "planted_attacks_200.jsonl"
$RunDir = "mitigation_results/planted200"
$BenignDir = "mitigation_results/planted30v2"   # benign n=200 already measured here

if (-not (Test-Path $Slice)) {
    Write-Error "Missing slice $Slice — build with: python build_planted_attacks.py --qa-jsonl data/question-answer/test.jsonl --n 200 --yesno-only --out $Slice"
}
if (-not (Test-Path "$RunDir/off.jsonl")) {
    Write-Error "Missing OFF arm $RunDir/off.jsonl — run check_off_asr.py first."
}

$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

function Invoke-Step([string]$Label, [string[]]$Args) {
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $py @Args
    if ($LASTEXITCODE -ne 0) { throw "$Label failed (exit $LASTEXITCODE)" }
}

# --- attack arms (reuse OFF) ---
Invoke-Step "full ON arm (n=200)" @(
    "run_mitigation_ab.py",
    "--slice", $Slice,
    "--detector", "existing",
    "--skip-off",
    "--run-dir", $RunDir,
    "--label", "full"
)

Invoke-Step "norefuse ON arm (n=200)" @(
    "run_mitigation_ab.py",
    "--slice", $Slice,
    "--detector", "existing",
    "--skip-off",
    "--no-refuse",
    "--run-dir", $RunDir,
    "--label", "norefuse"
)

# --- score ASR ---
Invoke-Step "score full" @(
    "score_mitigation_ab.py",
    "--off", "$RunDir/off.jsonl",
    "--on", "$RunDir/on_full.jsonl",
    "--out", "$RunDir/report_full.json"
)

Invoke-Step "score norefuse" @(
    "score_mitigation_ab.py",
    "--off", "$RunDir/off.jsonl",
    "--on", "$RunDir/on_norefuse.jsonl",
    "--out", "$RunDir/report_norefuse.json"
)

# --- benign trade-off (benign arms already n=200 in planted30v2) ---
Invoke-Step "benign cost vs full ASR (n=200)" @(
    "score_benign_cost.py",
    "--off", "$BenignDir/benign_off.jsonl",
    "--on", "$BenignDir/benign_on.jsonl",
    "--asr-report", "$RunDir/report_full.json",
    "--out", "$RunDir/benign_cost_report.json"
)

Invoke-Step "benign cost vs norefuse ASR (n=200)" @(
    "score_benign_cost.py",
    "--off", "$BenignDir/benign_off.jsonl",
    "--on", "$BenignDir/benign_on_norefuse.jsonl",
    "--asr-report", "$RunDir/report_norefuse.json",
    "--out", "$RunDir/benign_cost_norefuse.json"
)

Write-Host "`nDone. Thesis-safe trade-off table: $RunDir/benign_cost_norefuse.json" -ForegroundColor Green
Write-Host "  (benign n=200 from $BenignDir, ASR n=200 from $RunDir/report_norefuse.json)"
