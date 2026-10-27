param([switch]$DefinitionsOnly)

# Scheduling only: every scientific gate and audit remains in its frozen entry.
$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$SimPython = Join-Path $ProjectRoot '../.venv/Scripts/python.exe'
$BaselinePython = Join-Path $ProjectRoot '../.venv-baselines/Scripts/python.exe'
$Journal = Join-Path $ProjectRoot '.git/mainline-20261027-after-ood.jsonl'
$Recovery = 'artifacts/recovery_20261027/calibration_recovery'
$OodRoot = 'artifacts/recovery_20261026/ood_recovery'

function Write-Event([string]$Kind, $Detail) {
    $entry = [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); kind = $Kind; detail = $Detail }
    [IO.File]::AppendAllText($Journal, (($entry | ConvertTo-Json -Compress -Depth 8) + "`n"), [Text.UTF8Encoding]::new($false))
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    Write-Event 'command_start' @{ program = $Program; arguments = $Arguments }
    & $Program @Arguments
    $code = $LASTEXITCODE
    Write-Event 'command_exit' @{ program = $Program; arguments = $Arguments; exit_code = $code }
    if ($code -ne 0) { throw "Command failed with exit $code; retained output, no retry: $Program $Arguments" }
}

function Start-OwnedProcess([string]$Program, [string[]]$Arguments, [string]$Name) {
    $stdout = Join-Path $ProjectRoot ".git/mainline-20261027-$Name.stdout.log"
    $stderr = Join-Path $ProjectRoot ".git/mainline-20261027-$Name.stderr.log"
    if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'Owned job logs already exist; inspect instead of replacing' }
    $process = Start-Process -FilePath $Program -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $null = $process.Handle
    Write-Event 'owned_process_started' @{ role=$Name; pid=$process.Id; created_utc=$process.StartTime.ToUniversalTime().ToString('o'); arguments=$Arguments; stdout=$stdout; stderr=$stderr }
    return $process
}

function Wait-ActualZero($Process, [string]$Name) {
    while (-not $Process.WaitForExit(1000)) { }
    $code = $Process.ExitCode
    Write-Event 'owned_process_exit' @{ name = $Name; native_pid = $Process.Id; exit_code = $code }
    if ($null -eq $code -or $code -ne 0) { throw "Original $Name watcher failed or has no actual exit; preserve all data" }
    $Process.Dispose()
}

function Commit-Receipts([string]$Message, [string[]]$Directories) {
    # Explicit top-level JSON receipts/analysis only. Never stage live blocks,
    # unknown files, historical changes, ignored logs, or large raw databases.
    $paths = @()
    foreach ($directory in $Directories) {
        $full = Join-Path $ProjectRoot $directory
        if (-not (Test-Path -LiteralPath $full -PathType Container)) { throw "Missing result directory: $directory" }
        $paths += @(Get-ChildItem -LiteralPath $full -File -Filter '*.json' | ForEach-Object {
            $directory.Replace('\', '/').TrimEnd('/') + '/' + $_.Name
        })
    }
    if ($paths.Count -eq 0) { throw 'No receipts to commit' }
    & git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Index already contains changes; inspect instead of mixing commits' }
    Invoke-Checked 'git' (@('add', '--') + $paths)
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { throw 'Expected fresh result receipts; no automatic replay' }
    Invoke-Checked 'git' @('-c', 'maintenance.auto=false', '-c', 'gc.auto=0', 'commit', '--quiet', '-m', $Message)
}

function Invoke-FreshWatch([string]$Script, [string[]]$TailArguments, [string]$Execution) {
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $Execution)) {
        throw "Execution already exists; preserve prefix and inspect: $Execution"
    }
    Invoke-Checked $SimPython (@('-X', 'utf8', $Script, 'watch') + $TailArguments)
}

function Invoke-Mainline {
    Set-Location -LiteralPath $ProjectRoot
    if (Test-Path -LiteralPath $Journal) { throw 'Queue journal exists; inspect it, never auto-resume' }
    $env:PATH = (Join-Path $ProjectRoot '../.venv/Lib/site-packages/sumo/bin') + ';' + $env:PATH
    $env:PYTHONPATH = 'tools;src'
    $calibration = $null
    $ood = $null
    try {
        Write-Event 'queue_started' @{ protocol_changes = $false; retries = 0; shutdown = $false }
        # The first durable queue owns the still-running OOD review. Attach its
        # exact process now; never repeat its work or infer an exit from a file.
        $launch = Get-Content -LiteralPath (Join-Path $ProjectRoot '.git/durable_mainline_launch_20261027.json') -Raw | ConvertFrom-Json
        $priorQueue = [Diagnostics.Process]::GetProcessById($launch.process_id)
        $null = $priorQueue.Handle
        $expected = [DateTimeOffset]::Parse($launch.creation_date).UtcDateTime
        if ([Math]::Abs(($priorQueue.StartTime.ToUniversalTime() - $expected).TotalMilliseconds) -gt 2) { throw 'Prior queue identity differs' }
        Write-Event 'attached_prior_queue' @{ pid=$priorQueue.Id; created_utc=$priorQueue.StartTime.ToUniversalTime().ToString('o') }
        while (-not $priorQueue.WaitForExit(1000)) { }
        $priorCode = $priorQueue.ExitCode
        Write-Event 'prior_queue_exit' @{ exit_code=$priorCode }
        $priorQueue.Dispose()
        if ($priorCode -ne 1) { throw 'Unexpected prior queue exit; inspect without restarting anything' }
        $events = @(Get-Content -LiteralPath (Join-Path $ProjectRoot '.git/mainline-20261027.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
        $oodExit = @($events | Where-Object { $_.kind -eq 'owned_process_exit' -and $_.detail.name -eq 'ood_summary' -and $_.detail.exit_code -eq 0 })
        $calExit = @($events | Where-Object { $_.kind -eq 'owned_process_exit' -and $_.detail.name -eq 'calibration' -and $_.detail.exit_code -eq 1 })
        if ($oodExit.Count -ne 1 -or $calExit.Count -ne 1 -or $events[-1].kind -ne 'queue_stopped') { throw 'Expected complete OOD and pre-execution calibration refusal were not recorded' }
        if (Test-Path -LiteralPath (Join-Path $ProjectRoot "$Recovery/calibration/execution")) { throw 'Calibration prefix exists; no automatic recovery' }
        $failure = [IO.File]::ReadAllText((Join-Path $ProjectRoot '.git/mainline-20261027-calibration_watch.stderr.log'))
        if (-not $failure.Contains('Existing Python/SUMO process requires inspection before recovery')) { throw 'Unexpected calibration bootstrap failure' }
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "$OodRoot/analysis/report.json"))) { throw 'Full OOD report missing' }
        # Preserve the original quiet-project prerequisite before launching.
        $anchors = @($ProjectRoot.Replace('\','/').ToLower()+'/', ([IO.Path]::GetFullPath((Join-Path $ProjectRoot '../.venv'))).Replace('\','/').ToLower()+'/')
        do {
            $busy = @(Get-CimInstance Win32_Process | Where-Object {
                if ($_.Name -notmatch '^(python|sumo)') { return $false }
                $command = ($_.CommandLine+'').Replace('\','/').ToLower()
                return ($command.Contains($anchors[0]) -or $command.Contains($anchors[1]))
            })
            if ($busy.Count -gt 0) { Start-Sleep -Seconds 10 }
        } while ($busy.Count -gt 0)
        $calibration = Start-OwnedProcess $SimPython @('-X','utf8','tools/recover_prospective_20261027.py','watch','calibration') 'calibration_after_ood'
        Wait-ActualZero $calibration 'calibration'
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/recover_prospective_20261027.py', 'analyze', 'calibration')
        Commit-Receipts 'Complete recovered original calibration and fixed analysis' @("$Recovery/calibration/execution", "$Recovery/calibration/analysis")
        Invoke-FreshWatch 'tools/recover_prospective_20261027.py' @('confirmation') "$Recovery/confirmation/execution"
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/recover_prospective_20261027.py', 'analyze', 'confirmation')
        Commit-Receipts 'Complete original confirmation and retain fixed acceptance decision' @("$Recovery/confirmation/execution", "$Recovery/confirmation/analysis")

        $joint = 'artifacts/v42/joint_risk'
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/execute_joint_risk_recovered_20261027.py', 'bind')
        Commit-Receipts 'Bind unchanged joint risk study after complete predecessor' @($joint)
        foreach ($stage in @('calibration', 'confirmation')) {
            Invoke-FreshWatch 'tools/execute_joint_risk_recovered_20261027.py' @($stage) "$joint/$stage/execution"
            Commit-Receipts "Complete joint risk $stage native scope" @("$joint/$stage/execution")
        }
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/execute_joint_risk_recovered_20261027.py', 'finalize')
        Commit-Receipts 'Finalize full joint risk replay and fixed joint decision' @("$joint/analysis")

        $fair = 'artifacts/v44/fair_failure_preserving_execution'
        Invoke-FreshWatch 'tools/execute_fair_failure_v44.py' @() "$fair/execution"
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/execute_fair_failure_v44.py', 'summarize')
        Commit-Receipts 'Complete fair comparison with all failures preserved' @("$fair/execution", "$fair/analysis")

        $factorial = 'artifacts/candidate_factorial/complete_fair_cases'
        Invoke-FreshWatch 'tools/execute_candidate_factorial.py' @() "$factorial/execution"
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/finalize_episode_keys.py', 'factorial')
        Commit-Receipts 'Complete full candidate factorial and unchanged independent replay' @("$factorial/execution", "$factorial/analysis")
        $cspi = 'artifacts/literature_reproduction/cspi_factorial_prospective'
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/finalize_episode_keys.py', 'cspi-export')
        # Include the fixed .npz export needed by the separate baseline runtime.
        Invoke-Checked 'git' @('add', '--', "$cspi/data", "$cspi/key_export_compatibility.json")
        Invoke-Checked 'git' @('-c', 'maintenance.auto=false', '-c', 'gc.auto=0', 'commit', '--quiet', '-m', 'Export fixed prospective CSPI supplement after complete factorial')
        $source = Join-Path $ProjectRoot '../reference_sources/cspi-mt-b5af605/Figures_2_and_3_for_KDD.py'
        Invoke-Checked $BaselinePython @('-X', 'utf8', 'tools/finalize_episode_keys.py', 'cspi-analysis', '--source', $source)
        Commit-Receipts 'Complete original CSPI author tests and held out evaluation' @("$cspi/analysis")
        Invoke-Checked 'git' @('-c', 'maintenance.auto=false', '-c', 'gc.auto=0', '-c', 'http.proxy=http://127.0.0.1:7890', '-c', 'http.version=HTTP/1.1', '-c', 'http.postBuffer=67108864', '-c', 'http.lowSpeedLimit=1024', '-c', 'http.lowSpeedTime=60', 'push', 'origin', 'main')
        Write-Event 'queue_completed' @{ scope = 'fixed mainline collection and analyses'; raw_archive_delivery_and_paper_review_still_required = $true; research_success_not_assumed = $true }
    } catch {
        Write-Event 'queue_stopped' @{ reason = $_.Exception.Message; retry = $false; prefixes_preserved = $true }
        throw
    } finally {
        if ($null -ne $calibration) { $calibration.Dispose() }
        if ($null -ne $ood) { $ood.Dispose() }
    }
}

if (-not $DefinitionsOnly) { Invoke-Mainline }
