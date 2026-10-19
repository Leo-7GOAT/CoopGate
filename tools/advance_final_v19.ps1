<# Start the untouched final split after complete, audited calibration recovery.
Uses the exact native process, commit and publication guards already tested in
the original supervisor. No calibration worker is replaced or impersonated.
#>
[CmdletBinding()]
param(
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
$taskOriginalDriver = Join-Path $taskRepository 'tools/advance_validation_v19.ps1'
$taskOriginalSha = 'c68c67ce0d520ef536eb3f7e71c541bdd7ea041ae9cde08b2d164b7c56effab3'
if ((Get-FileHash -LiteralPath $taskOriginalDriver -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskOriginalSha) {
    throw 'The original tested process and archival guards changed.'
}
$taskTokens = $null
$taskErrors = $null
$taskAst = [System.Management.Automation.Language.Parser]::ParseFile($taskOriginalDriver, [ref]$taskTokens, [ref]$taskErrors)
if ($taskErrors.Count) { throw 'Original supervisor syntax is invalid.' }
$taskFunctions = $taskAst.FindAll({
    param($taskNode)
    $taskNode -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $taskNode.Name -in @('Write-TaskEvent','Invoke-TaskStep','Save-TaskCommit','Publish-TaskCommits','Wait-TaskWorkers')
}, $true)
if ($taskFunctions.Count -ne 5) { throw 'Expected original guard functions were not found.' }
foreach ($taskFunction in $taskFunctions) { . ([ScriptBlock]::Create($taskFunction.Extent.Text)) }

Push-Location -LiteralPath $taskRepository
try {
    foreach ($taskFuture in @('artifacts/v19/final_confirmation/summary.json','artifacts/v19/final_comparison',
                              'artifacts/v19/comparison_replay','artifacts/v19/final_report')) {
        if (Test-Path -LiteralPath $taskFuture) { throw "Existing final stage requires review: $taskFuture" }
    }
    if (Get-ChildItem -LiteralPath 'artifacts/v19/final_confirmation' -Directory -Filter 'shard_*') {
        throw 'Final workers already exist; never launch duplicate cases.'
    }
    $taskPrerequisites = @'
import json,sys
from pathlib import Path
sys.path.insert(0,'tools')
from start_final_v19 import check_launch,require_committed,verify_manifest
from probe_disturbed_gate_v19 import digest
audit=Path('artifacts/v19/risk_recovery_audit_20261019')
verify_manifest(audit)
require_committed([audit/'audit.json',audit/'sha256.json'])
record=json.loads((audit/'audit.json').read_text())
assert record['total_unique_study_runs']==40960
assert record['all_original_prefixes_and_traces_preserved'] and record['all_four_native_exits_normal']
assert not record['policy_or_statistical_rules_changed'] and not record['partial_risk_statistics_computed']
assert record['protocol_sha256']==digest('artifacts/v19/risk_validation/protocol.json')
for path,expected in record['recovery_runtime_evidence_sha256'].items():
    assert digest(path)==expected, path
check_launch()
print('Complete recovery and committed calibration/final-launch identities verified.')
'@
    & $taskPython -X utf8 -c $taskPrerequisites
    if ($LASTEXITCODE -ne 0) { throw 'Completed recovery and final-launch prerequisites failed.' }
    if ($CheckOnly) { Write-Output 'Read-only final prerequisites passed; no worker or output was created.'; return }
    & git ls-files --error-unmatch -- 'tools/advance_final_v19.ps1' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Commit the final continuation driver first.' }
    $taskChanges = & git status --porcelain -- 'tools/advance_final_v19.ps1'
    if ($LASTEXITCODE -ne 0 -or $taskChanges) { throw 'Final continuation driver must be committed and unchanged.' }
    if (Test-Path -LiteralPath $taskLogDirectory) { throw 'Preserve existing logs; use a new directory.' }
    $taskLockHandle = [System.IO.File]::Open($taskLock,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::Read)
    $taskLockBytes = [Text.Encoding]::UTF8.GetBytes([string]$PID)
    $taskLockHandle.Write($taskLockBytes,0,$taskLockBytes.Length)
    $taskLockHandle.Flush()
    New-Item -ItemType Directory -Path $taskLogDirectory -ErrorAction Stop | Out-Null
    $taskLogCreated = $true
    Write-TaskEvent -Phase 'started_after_completed_recovery' -Details @{
        supervisor_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        reused_guard_source_sha256=$taskOriginalSha
        goal_completion_claimed=$false
    }
    for ($taskShard=0; $taskShard -lt 4; $taskShard++) {
        $taskChild = Start-Process -FilePath $taskPython -WorkingDirectory $taskRepository -WindowStyle Hidden -PassThru `
            -ArgumentList @('-X','utf8','tools/start_final_v19.py','worker',[string]$taskShard) `
            -RedirectStandardOutput (Join-Path $taskLogDirectory "final_worker_$taskShard.stdout.log") `
            -RedirectStandardError (Join-Path $taskLogDirectory "final_worker_$taskShard.stderr.log")
        $null = $taskChild.Handle
        $taskOwnedWorkers += $taskChild
        Write-TaskEvent -Phase 'final_worker_started' -Details @{shard=$taskShard; worker_pid=$taskChild.Id}
    }
    Publish-TaskCommits 'publish_calibration_and_launch'
    Wait-TaskWorkers -Workers $taskOwnedWorkers -Study 'artifacts/v19/final_confirmation' -Phase 'waiting_final_confirmation'
    Invoke-TaskStep 'finalize_confirmation' $taskPython @('-X','utf8','tools/start_final_v19.py','finalize')
    Invoke-TaskStep 'evaluate_comparators' $taskPython @('-X','utf8','tools/compare_established_v19.py','evaluate',
        'artifacts/v19/comparison_plan','artifacts/v19/comparison_calibration','artifacts/v19/final_confirmation','artifacts/v19/final_comparison')
    Invoke-TaskStep 'replay_deployments' $taskPython @('-X','utf8','tools/replay_comparison_v19.py','run','artifacts/v19/comparison_replay')
    Invoke-TaskStep 'audit_final_report' $taskPython @('-X','utf8','tools/audit_final_report_v19.py','run','artifacts/v19/final_report')
    Invoke-TaskStep 'verify_preserved_evidence' $taskPython @('-X','utf8','tools/check_v19_evidence.py')
    Save-TaskCommit 'archive_final_results' @('artifacts/v19/final_confirmation','artifacts/v19/final_comparison',
        'artifacts/v19/comparison_replay','artifacts/v19/final_report') 'Archive independent V19 confirmation and verified whole-profile comparator results'
    Publish-TaskCommits 'publish_final_results'
    Write-TaskEvent -Phase 'completed' -Details @{publication_pending=$taskPublicationPending; goal_completion_claimed=$false}
} catch {
    if ($taskLogCreated) {
        Write-TaskEvent -Phase 'failed' -Details @{error=$_.Exception.Message; owned_final_worker_pids=@($taskOwnedWorkers | ForEach-Object { $_.Id }); goal_completion_claimed=$false}
    }
    throw
} finally {
    if ($null -ne $taskLockHandle) { $taskLockHandle.Dispose(); Remove-Item -LiteralPath $taskLock -ErrorAction SilentlyContinue }
    Pop-Location
}
