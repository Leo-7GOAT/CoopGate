param([switch]$DefinitionsOnly)

# Scheduling only: every scientific gate and audit remains in its frozen entry.
$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$SimPython = Join-Path $ProjectRoot '../.venv/Scripts/python.exe'
$BaselinePython = Join-Path $ProjectRoot '../.venv-baselines/Scripts/python.exe'
$Journal = Join-Path $ProjectRoot '.git/mainline-20261026.jsonl'
$Recovery = 'artifacts/recovery_20261026/calibration_recovery'
$Ood = 'artifacts/recovery_20261026/ood_recovery'

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

function Open-ExistingWatcher([int]$NativePid, [string]$Created, [string]$Receipt) {
    $launch = Get-Content -LiteralPath (Join-Path $ProjectRoot $Receipt) -Raw | ConvertFrom-Json
    if ($launch.watcher_pid -ne $NativePid) { throw 'Watcher PID differs from original launch receipt' }
    $process = [Diagnostics.Process]::GetProcessById($NativePid)
    # Access the handle NOW and retain it. Later PID reuse cannot supply an exit.
    $null = $process.Handle
    $expected = [DateTimeOffset]::Parse($Created).UtcDateTime
    if ([Math]::Abs(($process.StartTime.ToUniversalTime() - $expected).TotalMilliseconds) -gt 2) {
        $process.Dispose()
        throw 'Existing watcher creation identity differs; do not resume or relaunch'
    }
    Write-Event 'attached_native_handle' @{ native_pid = $NativePid; created_utc = $process.StartTime.ToUniversalTime().ToString('o'); receipt = $Receipt }
    return $process
}

function Wait-ActualZero($Process, [string]$Name) {
    while (-not $Process.WaitForExit(1000)) { }
    $code = $Process.ExitCode
    Write-Event 'attached_native_exit' @{ name = $Name; native_pid = $Process.Id; exit_code = $code }
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
        $calibration = Open-ExistingWatcher 52256 '2026-10-26T13:55:13.968914+08:00' "$Recovery/calibration/execution/watcher_launch.json"
        $ood = Open-ExistingWatcher 32664 '2026-10-26T13:59:47.583286+08:00' "$Ood/execution/spawn.json"

        Wait-ActualZero $ood 'ood'
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/recover_ood_20261026.py', 'summarize')
        Commit-Receipts 'Complete original OOD scope and full independent replay' @("$Ood/execution", "$Ood/analysis")

        Wait-ActualZero $calibration 'calibration'
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/recover_prospective_20261026.py', 'analyze', 'calibration')
        Commit-Receipts 'Complete recovered original calibration and fixed analysis' @("$Recovery/calibration/execution", "$Recovery/calibration/analysis")
        Invoke-FreshWatch 'tools/recover_prospective_20261026.py' @('confirmation') "$Recovery/confirmation/execution"
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/recover_prospective_20261026.py', 'analyze', 'confirmation')
        Commit-Receipts 'Complete original confirmation and retain fixed acceptance decision' @("$Recovery/confirmation/execution", "$Recovery/confirmation/analysis")

        $joint = 'artifacts/v42/joint_risk'
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/execute_joint_risk_recovered_20261026.py', 'bind')
        Commit-Receipts 'Bind unchanged joint risk study after complete predecessor' @($joint)
        foreach ($stage in @('calibration', 'confirmation')) {
            Invoke-FreshWatch 'tools/execute_joint_risk_recovered_20261026.py' @($stage) "$joint/$stage/execution"
            Commit-Receipts "Complete joint risk $stage native scope" @("$joint/$stage/execution")
        }
        Invoke-Checked $SimPython @('-X', 'utf8', 'tools/execute_joint_risk_recovered_20261026.py', 'finalize')
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
