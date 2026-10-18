<#
Continue the already-running, frozen V19 study. Never restart a worker, replace
a controller, tune a threshold, or inspect partial calibration outcomes.
All native command output is retained outside the immutable study artifacts.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int[]]$CalibrationProcessIds,
    [Parameter(Mandatory = $true)][string]$LogDirectory,
    [switch]$CheckOnly,
    [switch]$Publish
)

$ErrorActionPreference = 'Stop'
$taskRepository = [System.IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$taskPython = Join-Path (Split-Path $taskRepository -Parent) '.venv/Scripts/python.exe'
$taskLogDirectory = [System.IO.Path]::GetFullPath($LogDirectory)
$taskLock = Join-Path $taskRepository '.git/coopgate-v19-validation.lock'
$taskLockHandle = $null
$taskLogCreated = $false
$taskStepNumber = 0
$taskPublicationPending = $false
$taskOwnedWorkers = @()

function Write-TaskEvent {
    param([string]$Phase, [hashtable]$Details = @{})
    $taskEvent = @{utc = [DateTime]::UtcNow.ToString('o'); phase = $Phase; supervisor_pid = $PID}
    foreach ($taskKey in $Details.Keys) { $taskEvent[$taskKey] = $Details[$taskKey] }
    $taskLine = ConvertTo-Json -InputObject $taskEvent -Compress -Depth 8
    [System.IO.File]::AppendAllText((Join-Path $taskLogDirectory 'events.jsonl'), $taskLine + "`n", [System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::WriteAllText((Join-Path $taskLogDirectory 'state.json'), $taskLine, [System.Text.UTF8Encoding]::new($false))
    Write-Output $taskLine
}

function Invoke-TaskStep {
    param([string]$Name, [string]$Executable, [string[]]$CommandArguments)
    $script:taskStepNumber += 1
    $taskStepLog = Join-Path $taskLogDirectory ('{0:D2}_{1}.log' -f $script:taskStepNumber, $Name)
    Write-TaskEvent -Phase $Name -Details @{log = $taskStepLog}
    & $Executable @CommandArguments *> $taskStepLog
    if ($LASTEXITCODE -ne 0) { throw "Step $Name failed with exit code $LASTEXITCODE; inspect $taskStepLog" }
}

function Save-TaskCommit {
    param([string]$Name, [string[]]$Paths, [string]$Message)
    $taskAlreadyStaged = & git diff --cached --name-only
    if ($LASTEXITCODE -ne 0 -or $taskAlreadyStaged) { throw 'The index must be empty before archiving this experiment.' }
    Invoke-TaskStep -Name ($Name + '_stage') -Executable 'git' -CommandArguments (@('add', '--') + $Paths)
    Invoke-TaskStep -Name ($Name + '_whitespace') -Executable 'git' -CommandArguments @(
        '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check')
    Invoke-TaskStep -Name ($Name + '_commit') -Executable 'git' -CommandArguments @('commit', '-q', '-m', $Message)
}

function Publish-TaskCommits {
    param([string]$Name)
    if (-not $Publish) { $script:taskPublicationPending = $true; return }
    try {
        Invoke-TaskStep -Name $Name -Executable 'git' -CommandArguments @('-c', 'http.proxy=http://127.0.0.1:7890',
            '-c', 'http.lowSpeedLimit=1024', '-c', 'http.lowSpeedTime=60', 'push', 'origin', 'main')
        $script:taskPublicationPending = $false
    } catch {
        # Final-test preregistration requires local commits. A network failure
        # must not discard them, force-push, or idle the independent experiment.
        $script:taskPublicationPending = $true
        Write-TaskEvent -Phase 'publication_pending' -Details @{error = $_.Exception.Message}
    }
}

function Wait-TaskWorkers {
    param([System.Diagnostics.Process[]]$Workers, [string]$Study, [string]$Phase)
    while ($true) {
        $taskLiveCount = 0
        for ($taskIndex = 0; $taskIndex -lt $Workers.Count; $taskIndex++) {
            $taskWorker = $Workers[$taskIndex]
            $taskWorker.Refresh()
            if (-not $taskWorker.HasExited) { $taskLiveCount += 1; continue }
            $taskWorker.WaitForExit()
            if ($taskWorker.ExitCode -ne 0) { throw "Worker $($taskWorker.Id) exited with code $($taskWorker.ExitCode). No restart attempted." }
            if (-not (Test-Path -LiteralPath (Join-Path $taskRepository "$Study/shard_$taskIndex/complete.json"))) {
                throw "Worker $($taskWorker.Id) exited without its complete record. No restart attempted."
            }
        }
        Write-TaskEvent -Phase $Phase -Details @{live_workers = $taskLiveCount; worker_pids = @($Workers | ForEach-Object { $_.Id })}
        if ($taskLiveCount -eq 0) { return }
        Start-Sleep -Seconds 30
    }
}

Push-Location -LiteralPath $taskRepository
try {
    if ($CalibrationProcessIds.Count -ne 4 -or @($CalibrationProcessIds | Select-Object -Unique).Count -ne 4) {
        throw 'Four distinct current calibration process IDs in shard 0..3 order are required.'
    }
    if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Research Python runtime is missing.' }
    foreach ($taskFuturePath in @('artifacts/v19/risk_validation/summary.json', 'artifacts/v19/comparison_calibration',
                                  'artifacts/v19/final_launch.json', 'artifacts/v19/final_comparison',
                                  'artifacts/v19/comparison_replay', 'artifacts/v19/final_report')) {
        if (Test-Path -LiteralPath $taskFuturePath) { throw "Existing stage $taskFuturePath requires explicit review, not a rerun." }
    }
    if (Get-ChildItem -LiteralPath 'artifacts/v19/final_confirmation' -Directory -Filter 'shard_*') {
        throw 'Independent final workers already exist; never launch a duplicate.'
    }
    $taskCalibrationWorkers = @()
    foreach ($taskId in $CalibrationProcessIds) {
        $taskProcess = Get-Process -Id $taskId -ErrorAction Stop
        # Open and retain a handle to this exact process; PID reuse cannot
        # silently make the watcher follow a different process later.
        $null = $taskProcess.Handle
        if ($taskProcess.HasExited) { throw "Calibration process $taskId is no longer live." }
        $taskCalibrationWorkers += $taskProcess
    }
    $taskValidationCode = @'
import sys
from pathlib import Path
sys.path.insert(0, 'tools')
from study_gate_v19 import read_protocol
from compare_established_v19 import read_plan
from start_final_v19 import verify_manifest
for name in ('risk_validation', 'final_confirmation'):
    read_protocol(Path('artifacts/v19') / name)
read_plan('artifacts/v19/comparison_plan')
verify_manifest('artifacts/v19/final_report_plan')
verify_manifest('artifacts/v19/comparison_replay_plan')
print('Frozen sources and plans verified; no outcome records read.')
'@
    & $taskPython -X utf8 -c $taskValidationCode
    if ($LASTEXITCODE -ne 0) { throw 'Frozen-source prerequisite check failed.' }
    if ($CheckOnly) {
        Write-Output 'Read-only prerequisite check passed. No log directory, stage, worker or commit was created.'
        return
    }
    & git ls-files --error-unmatch -- 'tools/advance_validation_v19.ps1' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Commit the supervisor before running it.' }
    $taskSupervisorChanges = & git status --porcelain -- 'tools/advance_validation_v19.ps1'
    if ($LASTEXITCODE -ne 0 -or $taskSupervisorChanges) { throw 'The supervisor must be committed and unchanged.' }
    if (Test-Path -LiteralPath $taskLogDirectory) { throw 'Use a new log directory; preserve earlier evidence.' }
    $taskLockHandle = [System.IO.File]::Open($taskLock, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::Read)
    $taskLockBytes = [System.Text.Encoding]::UTF8.GetBytes([string]$PID)
    $taskLockHandle.Write($taskLockBytes, 0, $taskLockBytes.Length)
    $taskLockHandle.Flush()
    New-Item -ItemType Directory -Path $taskLogDirectory -ErrorAction Stop | Out-Null
    $taskLogCreated = $true
    Write-TaskEvent -Phase 'started' -Details @{
        calibration_pids = $CalibrationProcessIds
        supervisor_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        study_selection = 'Frozen protocol; negative certification results retained'
    }
    Wait-TaskWorkers -Workers $taskCalibrationWorkers -Study 'artifacts/v19/risk_validation' -Phase 'waiting_calibration'
    Invoke-TaskStep 'finalize_calibration' $taskPython @('-X', 'utf8', 'tools/study_gate_v19.py', 'finalize', 'artifacts/v19/risk_validation')
    Invoke-TaskStep 'calibrate_adapters' $taskPython @('-X', 'utf8', 'tools/compare_established_v19.py', 'calibrate',
        'artifacts/v19/comparison_plan', 'artifacts/v19/risk_validation', 'artifacts/v19/comparison_calibration')
    Save-TaskCommit 'archive_calibration' @('artifacts/v19/risk_validation', 'artifacts/v19/comparison_calibration') `
        'Archive complete V19 conditional-risk calibration without selecting a training seed'
    Invoke-TaskStep 'prepare_final' $taskPython @('-X', 'utf8', 'tools/start_final_v19.py', 'prepare')
    Save-TaskCommit 'archive_launch' @('artifacts/v19/final_launch.json') 'Freeze V19 independent confirmation launch after completed calibration'
    for ($taskShard = 0; $taskShard -lt 4; $taskShard++) {
        $taskChild = Start-Process -FilePath $taskPython -WorkingDirectory $taskRepository -WindowStyle Hidden -PassThru `
            -ArgumentList @('-X', 'utf8', 'tools/start_final_v19.py', 'worker', [string]$taskShard) `
            -RedirectStandardOutput (Join-Path $taskLogDirectory "final_worker_$taskShard.stdout.log") `
            -RedirectStandardError (Join-Path $taskLogDirectory "final_worker_$taskShard.stderr.log")
        $taskOwnedWorkers += $taskChild
        Write-TaskEvent -Phase 'final_worker_started' -Details @{shard = $taskShard; worker_pid = $taskChild.Id}
    }
    Publish-TaskCommits 'publish_calibration_and_launch'
    Wait-TaskWorkers -Workers $taskOwnedWorkers -Study 'artifacts/v19/final_confirmation' -Phase 'waiting_final_confirmation'
    Invoke-TaskStep 'finalize_confirmation' $taskPython @('-X', 'utf8', 'tools/start_final_v19.py', 'finalize')
    Invoke-TaskStep 'evaluate_comparators' $taskPython @('-X', 'utf8', 'tools/compare_established_v19.py', 'evaluate',
        'artifacts/v19/comparison_plan', 'artifacts/v19/comparison_calibration', 'artifacts/v19/final_confirmation', 'artifacts/v19/final_comparison')
    Invoke-TaskStep 'replay_deployments' $taskPython @('-X', 'utf8', 'tools/replay_comparison_v19.py', 'run', 'artifacts/v19/comparison_replay')
    Invoke-TaskStep 'audit_final_report' $taskPython @('-X', 'utf8', 'tools/audit_final_report_v19.py', 'run', 'artifacts/v19/final_report')
    Invoke-TaskStep 'verify_preserved_evidence' $taskPython @('-X', 'utf8', 'tools/check_v19_evidence.py')
    Save-TaskCommit 'archive_final_results' @('artifacts/v19/final_confirmation', 'artifacts/v19/final_comparison',
        'artifacts/v19/comparison_replay', 'artifacts/v19/final_report') 'Archive independent V19 confirmation and verified whole-profile comparator results'
    Publish-TaskCommits 'publish_final_results'
    Write-TaskEvent -Phase 'completed' -Details @{publication_pending = $taskPublicationPending; goal_completion_claimed = $false}
} catch {
    if ($taskLogCreated) {
        Write-TaskEvent -Phase 'failed' -Details @{error = $_.Exception.Message; owned_final_worker_pids = @($taskOwnedWorkers | ForEach-Object { $_.Id }); goal_completion_claimed = $false}
    }
    throw
} finally {
    if ($null -ne $taskLockHandle) {
        $taskLockHandle.Dispose()
        Remove-Item -LiteralPath $taskLock -ErrorAction SilentlyContinue
    }
    Pop-Location
}
