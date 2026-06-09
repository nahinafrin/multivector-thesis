# run_with_keepawake.ps1 - launches the QA pipeline and asks Windows
# to keep the system awake while the python process is alive.
#
# Run detached, e.g.:
#   Start-Process powershell -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","run_with_keepawake.ps1" -WindowStyle Hidden -WorkingDirectory $PWD

Set-Location -Path $PSScriptRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8       = "1"

Add-Type -Namespace KeepAwake -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError=true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@

$ES_CONTINUOUS       = [uint32]0x80000000
$ES_SYSTEM_REQUIRED  = [uint32]0x00000001
$ES_AWAYMODE_REQUIRED= [uint32]0x00000040
[void][KeepAwake.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED)

try {
    $proc = Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\python.exe" `
        -ArgumentList @(
            "kb_rag_mini_wikipedia.py",
            "--all","--resume",
            "--index","./kb_wiki",
            "--k","5","--top-n","3",
            "--results","results.jsonl"
        ) `
        -WorkingDirectory $PWD `
        -RedirectStandardOutput run_kb_full.log `
        -RedirectStandardError run_kb_full.err `
        -WindowStyle Hidden -PassThru
    $proc.Id | Out-File run_kb.pid -Encoding ascii
    Wait-Process -Id $proc.Id
} finally {
    [void][KeepAwake.Power]::SetThreadExecutionState($ES_CONTINUOUS)
}
