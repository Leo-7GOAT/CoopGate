<# Exercise process-exit/complete-record guards using actual short-lived processes.
No simulator, calibration outcome, Git mutation or final worker is used.
#>
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$taskSourceRoot = Split-Path $PSScriptRoot -Parent
$taskDriverSource = Join-Path $taskSourceRoot 'tools/advance_validation_v19.ps1'
$taskTestPython = Join-Path (Split-Path $taskSourceRoot -Parent) '.venv/Scripts/python.exe'
$taskTokens = $null
$taskErrors = $null
$taskAst = [System.Management.Automation.Language.Parser]::ParseFile($taskDriverSource, [ref]$taskTokens, [ref]$taskErrors)
if ($taskErrors.Count) { throw 'Driver syntax is invalid.' }
$taskFunctions = $taskAst.FindAll({
    param($taskNode)
    $taskNode -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $taskNode.Name -in @('Write-TaskEvent', 'Wait-TaskWorkers')
}, $true)
if ($taskFunctions.Count -ne 2) { throw 'Expected guard functions were not found.' }
foreach ($taskFunction in $taskFunctions) {
    . ([ScriptBlock]::Create($taskFunction.Extent.Text))
}
$taskTestRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('coopgate-driver-test-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskTestRoot | Out-Null
$taskChecks = @()

foreach ($taskCase in @(
    @{name = 'normal_completed_exit'; code = 0; complete = $true; error = $null},
    @{name = 'missing_completion'; code = 0; complete = $false; error = 'without its complete record'},
    @{name = 'nonzero_exit_despite_completion'; code = 7; complete = $true; error = 'exited with code 7'}
)) {
    $taskRepository = Join-Path $taskTestRoot $taskCase.name
    $taskLogDirectory = Join-Path $taskRepository 'logs'
    New-Item -ItemType Directory -Path (Join-Path $taskRepository 'study/shard_0') -Force | Out-Null
    New-Item -ItemType Directory -Path $taskLogDirectory | Out-Null
    if ($taskCase.complete) {
        [System.IO.File]::WriteAllText((Join-Path $taskRepository 'study/shard_0/complete.json'), '{}')
    }
    $taskChildScript = Join-Path $taskRepository 'short_worker.py'
    [System.IO.File]::WriteAllText($taskChildScript, "import time`ntime.sleep(0.5)`nraise SystemExit($($taskCase.code))`n")
    $taskStarted = Start-Process -FilePath $taskTestPython -WindowStyle Hidden -PassThru `
        -ArgumentList @('-X', 'utf8', ('"' + $taskChildScript + '"'))
    # Reattach exactly as the driver does for existing calibration workers.
    $taskAttached = Get-Process -Id $taskStarted.Id
    $null = $taskAttached.Handle
    if (-not $taskAttached.WaitForExit(10000)) { throw 'The short test process did not exit.' }
    $taskFailure = $null
    try {
        Wait-TaskWorkers -Workers @($taskAttached) -Study 'study' -Phase 'test_wait'
    } catch {
        $taskFailure = $_.Exception.Message
    }
    if ($null -eq $taskCase.error) {
        if ($null -ne $taskFailure) { throw "Normal worker was rejected: $taskFailure" }
    } elseif ($null -eq $taskFailure -or -not $taskFailure.Contains($taskCase.error)) {
        throw "Expected $($taskCase.error), received $taskFailure"
    }
    $taskChecks += @{case = $taskCase.name; passed = $true; observed_exit_code = $taskAttached.ExitCode; restarted = $false}
    $taskStarted.Dispose()
    $taskAttached.Dispose()
}
$taskResult = @{
    purpose = 'Windows native-process lifetime and completion guards'
    source_sha256 = (Get-FileHash -LiteralPath $taskDriverSource -Algorithm SHA256).Hash.ToLowerInvariant()
    checks = $taskChecks
    all_passed = $true
    calibration_outcomes_read = $false
    additional_sumo_runs = 0
}
$taskResultFile = Join-Path $taskTestRoot 'result.json'
$taskResult | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $taskResultFile -Encoding utf8
Write-Output "Native process checks passed: $taskResultFile"
