$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'advance_mainline_20261027.ps1') -DefinitionsOnly
$tag = [Guid]::NewGuid().ToString('N')
$Journal = Join-Path $ProjectRoot ".git/durable-queue-test-$tag.jsonl"
$checks = 0
function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
foreach ($code in @(0, 7)) {
    $name = "test-$tag-$code"
    $child = Start-OwnedProcess 'powershell.exe' @('-NoProfile','-Command',"Start-Sleep -Seconds 2; exit $code") $name
    $rejected = $false
    try { Wait-ActualZero $child $name } catch { $rejected = $true }
    Require ($rejected -eq ($code -ne 0)) 'Actual child failure was not preserved'
    $events = @(Get-Content -LiteralPath $Journal | ForEach-Object { $_ | ConvertFrom-Json })
    Require ($events[-1].detail.exit_code -eq $code) 'Actual child exit code was lost'
    $rejected = $false
    try { $null = Start-OwnedProcess 'powershell.exe' @('-NoProfile','-Command','exit 99') $name } catch { $rejected = $true }
    Require $rejected 'Existing logs were overwritten or an owned job was restarted'
}
$rejected = $false
try { Invoke-Mainline } catch { $rejected = $true }
Require $rejected 'Existing queue journal was resumed implicitly'
"$checks durable queue checks passed; no scientific job was launched"
