param([switch]$InspectOnly)
$ErrorActionPreference = 'Stop'
$repoDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$chainDir = Join-Path $repoDir 'artifacts/v29/recovery_chain'
$planPath = Join-Path $chainDir 'plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
function Read-Json([string]$path) { Get-Content -LiteralPath $path -Raw | ConvertFrom-Json }
function Digest([string]$path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() }
function Save-Json([string]$path, $value) {
    $stream = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
        $writer.WriteLine(($value | ConvertTo-Json -Depth 10))
        $writer.Flush()
    } finally { $stream.Dispose() }
}
foreach ($file in $plan.requiredFiles.PSObject.Properties) {
    if ((Digest (Join-Path $repoDir $file.Name)) -ne $file.Value) { throw "Frozen dependency changed: $($file.Name)" }
}
$handles = @()
foreach ($identity in $plan.dependencies) {
    $process = Get-Process -Id $identity.pid -ErrorAction Stop
    $expectedStart = ([DateTime]$identity.createdUtc).ToUniversalTime().ToString('o')
    if ($process.ProcessName -ne 'python' -or $process.StartTime.ToUniversalTime().ToString('o') -ne $expectedStart) {
        throw 'PID reuse or a different native process; do not infer completion'
    }
    $null = $process.Handle
    $handles += @{process=$process;identity=$identity;done=$false}
}
if ($InspectOnly) {
    @{verifiedLiveDependencies=$handles.Count;planSha256=(Digest $planPath);noExecutionStarted=$true} | ConvertTo-Json -Compress
    exit 0
}
$execution = Join-Path $chainDir 'execution'
$null = New-Item -ItemType Directory -Path $execution -Force
Save-Json (Join-Path $execution 'start.json') @{utc=[DateTime]::UtcNow.ToString('o');pid=$PID;planSha256=(Digest $planPath)}
$exits = @()
$lastUpdate = [DateTime]::MinValue
while (@($handles | Where-Object { -not $_.done }).Count -gt 0) {
    foreach ($item in $handles) {
        if (-not $item.done -and $item.process.WaitForExit(1000)) {
            $item.done = $true
            $exits += @{pid=$item.identity.pid;createdUtc=$item.identity.createdUtc;role=$item.identity.role;exitCode=$item.process.ExitCode}
        }
    }
    if (([DateTime]::UtcNow-$lastUpdate).TotalSeconds -ge 30) {
        @{utc=[DateTime]::UtcNow.ToString('o');phase='waiting_for_buffered_exit';remaining=@($handles | Where-Object {-not $_.done} | ForEach-Object {$_.identity.pid});observedExits=$exits} |
            ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $execution 'state.json') -Encoding utf8
        $lastUpdate=[DateTime]::UtcNow
    }
}
Save-Json (Join-Path $execution 'dependencies_exited.json') @{utc=[DateTime]::UtcNow.ToString('o');exits=$exits}
if (@($exits | Where-Object { $_.exitCode -ne 0 }).Count) { throw 'A real dependency exited unsuccessfully; calibration remains unstarted' }
$bufferDir = Join-Path $repoDir $plan.bufferDirectory
$exit = Read-Json (Join-Path $bufferDir 'execution/exit.json')
$done = Read-Json (Join-Path $bufferDir 'execution/collection_complete.json')
$review = Read-Json (Join-Path $bufferDir 'execution/all_blocks_reviewed.json')
$boundary = Read-Json (Join-Path $bufferDir 'boundary/00226/verified.json')
if ($exit.exit_code -ne 0 -or $exit.native_pid -ne $plan.bufferWorkerPid -or $exit.launcher_pid -ne $plan.bufferWorkerLauncherPid -or
    $done.native_pid -ne $plan.bufferWorkerPid -or $done.actual_sumo_runs -ne 6912 -or $done.blocks -ne 288 -or
    $done.retained_roles -ne 5435 -or $done.new_roles -ne 1477 -or $done.boundary_replays -ne 1 -or
    $done.plan_sha256 -ne (Digest (Join-Path $bufferDir 'plan.json')) -or
    $exit.plan_sha256 -ne $done.plan_sha256 -or
    $done.recovery_plan_sha256 -ne (Digest (Join-Path $bufferDir 'recovery/plan.json')) -or
    $done.fresh_review_sha256 -ne (Digest (Join-Path $bufferDir 'execution/all_blocks_reviewed.json')) -or
    $review.reviewed_roles -ne 6912 -or $review.paired_cases -ne 3456 -or @($review.blocks).Count -ne 288 -or
    @($review.blocks | Where-Object { -not $_.all_original_and_pair_checks_pass }).Count -ne 0 -or
    -not $boundary.exact_non_timing_match -or $boundary.actual_sumo_runs -ne 1) {
    throw 'Complete buffered scope and real success receipts are required'
}
$calDir = Join-Path $repoDir $plan.calibrationDirectory
if (Test-Path -LiteralPath (Join-Path $calDir 'calibration/execution')) { throw 'Calibration has execution evidence; never launch a duplicate' }
foreach ($file in $plan.requiredFiles.PSObject.Properties) {
    if ((Digest (Join-Path $repoDir $file.Name)) -ne $file.Value) { throw 'A frozen launch dependency changed while waiting' }
}
$stdout = Join-Path $calDir 'launcher_after_buffer.stdout.log'
$stderr = Join-Path $calDir 'launcher_after_buffer.stderr.log'
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'Preserve an existing launch log' }
Save-Json (Join-Path $execution 'calibration_launch_requested.json') @{utc=[DateTime]::UtcNow.ToString('o');priorLaunchExit='unknown; guard rejected before execution';bufferExitSha256=(Digest (Join-Path $bufferDir 'execution/exit.json'))}
$calProcess = Start-Process -FilePath $plan.python -ArgumentList @('-X','utf8','tools/recover_prospective_v29.py','watch','calibration') -WorkingDirectory $repoDir -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Save-Json (Join-Path $execution 'calibration_launcher.json') @{utc=[DateTime]::UtcNow.ToString('o');launcherPid=$calProcess.Id;createdUtc=$calProcess.StartTime.ToUniversalTime().ToString('o')}
@{utc=[DateTime]::UtcNow.ToString('o');phase='calibration_launched';launcherPid=$calProcess.Id} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $execution 'state.json') -Encoding utf8
$calProcess.WaitForExit()
Save-Json (Join-Path $execution 'calibration_launcher_exit.json') @{utc=[DateTime]::UtcNow.ToString('o');launcherPid=$calProcess.Id;exitCode=$calProcess.ExitCode}
if ($calProcess.ExitCode -ne 0) { throw 'Calibration launcher failed; preserve outputs and inspect the actual state' }
