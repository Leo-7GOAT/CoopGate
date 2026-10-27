param([switch]$DefinitionsOnly)
$ErrorActionPreference='Stop'
$OnlyDefinitionsRequested=[bool]$DefinitionsOnly
. (Join-Path $PSScriptRoot 'continue_mainline_20261027_v2.ps1') -DefinitionsOnly
$Journal=Join-Path $ProjectRoot '.git/mainline-after-authorized-withdrawal-20261027.jsonl'

function Invoke-WithdrawnMainline {
    Set-Location -LiteralPath $ProjectRoot
    if(Test-Path -LiteralPath $Journal){throw 'Queue exists; preserve it, never auto-resume'}
    $env:PATH=(Join-Path $ProjectRoot '../.venv/Lib/site-packages/sumo/bin')+';'+$env:PATH
    $env:PYTHONPATH='tools;src'
    $route='tools/execute_joint_risk_after_withdrawal_20261027.py'
    try {
        Write-Event 'queue_started' @{authorized_prerequisite_withdrawal=$true;legacy_completed=$false;new_runs=16384;threshold_changes=$false;retries=0;shutdown=$false}
        Invoke-Checked $SimPython @('-X','utf8','tools/legacy_withdrawal_contract_20261027.py','verify')
        $joint='artifacts/v42/joint_risk'
        Invoke-Checked $SimPython @('-X','utf8',$route,'bind')
        Commit-Receipts 'Bind full joint risk after authorized incomplete legacy withdrawal' @($joint)
        foreach($stage in @('calibration','confirmation')){
            Invoke-FreshWatch $route @($stage) "$joint/$stage/execution"
            Commit-Receipts "Complete unchanged joint risk $stage after authorized withdrawal" @("$joint/$stage/execution")
        }
        Invoke-Checked $SimPython @('-X','utf8',$route,'finalize')
        Commit-Receipts 'Finalize full joint risk with original tests and independent replay' @("$joint/analysis")
        $fair='artifacts/v44/fair_failure_preserving_execution'
        Invoke-FreshWatch 'tools/execute_fair_failure_v44.py' @() "$fair/execution"
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
        Invoke-Checked 'git' @('-c','maintenance.auto=false','-c','gc.auto=0','-c','http.proxy=http://127.0.0.1:7890','-c','http.version=HTTP/1.1','-c','http.postBuffer=67108864','-c','http.lowSpeedLimit=1024','-c','http.lowSpeedTime=60','push','origin','main')
        Write-Event 'queue_completed' @{scope='full original joint, fair, factorial, CSPI';legacy_status='withdrawn incomplete';raw_delivery_and_paper_review_still_required=$true;research_success_not_assumed=$true}
    }catch{
        Write-Event 'queue_stopped' @{reason=$_.Exception.Message;retry=$false;prefixes_preserved=$true}
        throw
    }
}

if(-not $OnlyDefinitionsRequested){Invoke-WithdrawnMainline}
