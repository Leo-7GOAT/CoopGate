param([switch]$DefinitionsOnly)
$ErrorActionPreference='Stop'
$OnlyDefinitionsRequested=[bool]$DefinitionsOnly
. (Join-Path $PSScriptRoot 'continue_mainline_20261027_v2.ps1') -DefinitionsOnly
$Journal=Join-Path $ProjectRoot '.git/mainline-fair-replay-20261101.jsonl'
function Invoke-FairReplayRecovery {
    Set-Location -LiteralPath $ProjectRoot
    if(Test-Path -LiteralPath $Journal){throw 'Preserve existing recovery journal; no implicit resume'}
    $busy=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'execute_fair_failure_v44|joint_recovery_20261031|execute_candidate_factorial|finalize_episode_keys'})
    if($busy.Count){throw 'Existing study process remains live; no duplicate work'}
    $env:PATH=(Join-Path $ProjectRoot '../.venv/Lib/site-packages/sumo/bin')+';'+$env:PATH
    $env:PYTHONPATH='tools;src'
    Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class CoopGateWakeNov1 { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
    if([CoopGateWakeNov1]::SetThreadExecutionState([uint32]2147483649) -eq 0){throw 'Power request failed'}
    try {
        Write-Event 'queue_started' @{original_summary_exit='unknown';fair_collection_complete=$true;new_fair_runs=0;threshold_changes=$false;automatic_retry=$false;shutdown=$false}
        Invoke-Checked 'C:/Users/Leo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' @('-X','utf8','operations/verify_fair_replay_recovery_20261101.py','verify')
        $fair='artifacts/v44/fair_failure_preserving_execution'
        # Collection is complete and sealed; repeat only the full original review.
        Invoke-Checked $SimPython @('-X','utf8','tools/execute_fair_failure_v44.py','summarize')
        Commit-Receipts 'Complete all fair selectors with failures preserved' @("$fair/execution","$fair/analysis")
        $factorial='artifacts/candidate_factorial/complete_fair_cases'
        Invoke-FreshWatch 'tools/execute_candidate_factorial.py' @() "$factorial/execution"
        Invoke-Checked $SimPython @('-X','utf8','tools/finalize_episode_keys.py','factorial')
        Commit-Receipts 'Complete unchanged candidate factorial and full replay' @("$factorial/execution","$factorial/analysis")
        $cspi='artifacts/literature_reproduction/cspi_factorial_prospective'
        Invoke-Checked $SimPython @('-X','utf8','tools/finalize_episode_keys.py','cspi-export')
        Invoke-Checked 'git' @('add','--',"$cspi/data","$cspi/key_export_compatibility.json")
        Invoke-Checked 'git' @('-c','maintenance.auto=false','-c','gc.auto=0','commit','--quiet','-m','Export full prospective CSPI data after complete factorial')
        $source=Join-Path $ProjectRoot '../reference_sources/cspi-mt-b5af605/Figures_2_and_3_for_KDD.py'
        Invoke-Checked $BaselinePython @('-X','utf8','tools/finalize_episode_keys.py','cspi-analysis','--source',$source)
        Commit-Receipts 'Complete original author CSPI tests and held out evaluation' @("$cspi/analysis")
        Write-Event 'queue_completed' @{scope='full original joint, fair, factorial, CSPI';legacy_status='withdrawn incomplete';raw_delivery_and_paper_review_still_required=$true;research_success_not_assumed=$true}
    }catch{
        Write-Event 'queue_stopped' @{reason=$_.Exception.Message;retry=$false;prefixes_preserved=$true}
        throw
    }finally{
        [void][CoopGateWakeNov1]::SetThreadExecutionState([uint32]2147483648)
    }
}
if(-not $OnlyDefinitionsRequested){Invoke-FairReplayRecovery}
