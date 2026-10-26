$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'advance_mainline_20261026.ps1') -DefinitionsOnly
$tag = [Guid]::NewGuid().ToString('N')
$Journal = Join-Path $ProjectRoot ".git/queue-test-$tag.jsonl"
$receiptRelative = ".git/queue-test-$tag-receipt.json"
$receipt = Join-Path $ProjectRoot $receiptRelative
$processes = @()
$checks = 0
function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
try {
    Set-Location -LiteralPath $ProjectRoot
    # Attach through a fresh GetProcessById object, matching production use.
    $child = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-Command', 'Start-Sleep -Seconds 3; exit 0') -WindowStyle Hidden -PassThru
    $processes += $child
    @{ watcher_pid = $child.Id } | ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding UTF8
    $created = $child.StartTime.ToUniversalTime().ToString('o')
    $attached = Open-ExistingWatcher $child.Id $created $receiptRelative
    Wait-ActualZero $attached 'test-zero'
    Require $child.WaitForExit(1000) 'Wait returned before the actual native exit'
    $events = @(Get-Content -LiteralPath $Journal | ForEach-Object { $_ | ConvertFrom-Json })
    Require ($events[-1].detail.exit_code -eq 0) 'Actual zero exit was not recorded'

    $child = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-Command', 'Start-Sleep -Seconds 3; exit 7') -WindowStyle Hidden -PassThru
    $processes += $child
    @{ watcher_pid = $child.Id } | ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding UTF8
    $created = $child.StartTime.ToUniversalTime().ToString('o')
    $attached = Open-ExistingWatcher $child.Id $created $receiptRelative
    $rejected = $false
    try { Wait-ActualZero $attached 'test-failure' } catch { $rejected = $true }
    Require $rejected 'Nonzero watcher exit was accepted'
    $events = @(Get-Content -LiteralPath $Journal | ForEach-Object { $_ | ConvertFrom-Json })
    Require ($events[-1].detail.exit_code -eq 7) 'Nonzero native code was replaced or lost'

    @{ watcher_pid = $PID } | ConvertTo-Json | Set-Content -LiteralPath $receipt -Encoding UTF8
    $rejected = $false
    try { $null = Open-ExistingWatcher $PID '2000-01-01T00:00:00Z' $receiptRelative } catch { $rejected = $true }
    Require $rejected 'A reused/mismatched creation identity was accepted'
    $rejected = $false
    try { $null = Open-ExistingWatcher ($PID + 1) '2000-01-01T00:00:00Z' $receiptRelative } catch { $rejected = $true }
    Require $rejected 'Mismatched launch receipt was accepted'

    $followed = $false
    try {
        Invoke-Checked 'powershell.exe' @('-NoProfile', '-Command', 'exit 9')
        $followed = $true
    } catch { }
    Require (-not $followed) 'Failed command did not stop the chain'
    $rejected = $false
    try { Invoke-FreshWatch 'must-never-execute.py' @() '.git' } catch { $rejected = $true }
    Require $rejected 'An existing execution could be restarted'
    $rejected = $false
    try { Invoke-Mainline } catch { $rejected = $true }
    Require $rejected 'An existing journal could be resumed implicitly'
    Write-Output "$checks queue checks passed; no simulation or frozen source changed"
} finally {
    foreach ($process in $processes) {
        if (-not $process.HasExited) { $process.WaitForExit() }
        $process.Dispose()
    }
    foreach ($path in @($Journal, $receipt)) {
        if (Test-Path -LiteralPath $path -PathType Leaf) { Remove-Item -LiteralPath $path }
    }
}
