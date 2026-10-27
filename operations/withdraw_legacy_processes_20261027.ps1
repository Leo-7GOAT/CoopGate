$ErrorActionPreference='Stop'
$Root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if($Root -ne 'C:\Users\Leo\Desktop\papper\CoopGate'){throw 'Unexpected project root'}
Set-Location -LiteralPath $Root
$Out=Join-Path $Root 'artifacts/legacy_withdrawal_20261027'
if(Test-Path -LiteralPath $Out){throw 'Withdrawal evidence exists; inspect, never repeat'}
$Expected=@{
  58280='2026-10-27T08:35:19.457505+08:00'
  26456='2026-10-27T13:30:11.881221+08:00'
  42648='2026-10-27T13:33:02.257675+08:00'
  71328='2026-10-27T13:33:06.946703+08:00'
  64372='2026-10-27T13:33:11.621409+08:00'
}
$All=@(Get-CimInstance Win32_Process)
foreach($id in $Expected.Keys){
  $p=@($All|Where-Object ProcessId -eq $id)
  if($p.Count -ne 1 -or [Math]::Abs((([DateTimeOffset]$p[0].CreationDate).UtcDateTime-([DateTimeOffset]::Parse($Expected[$id])).UtcDateTime).TotalMilliseconds) -gt 2){throw "Owned process identity differs: $id"}
}
$queue=@($All|Where-Object ProcessId -eq 58280)[0]
if(-not $queue.CommandLine.Contains((Join-Path $Root 'operations\continue_mainline_20261027_v2.ps1'))){throw 'Queue command differs'}
$Depth=@{58280=0}
do{
  $before=$Depth.Count
  foreach($p in $All){if($Depth.ContainsKey([int]$p.ParentProcessId) -and -not $Depth.ContainsKey([int]$p.ProcessId)){$Depth[[int]$p.ProcessId]=$Depth[[int]$p.ParentProcessId]+1}}
}while($Depth.Count -ne $before)
$Owned=@($All|Where-Object{$Depth.ContainsKey([int]$_.ProcessId)}|Sort-Object @{Expression={$Depth[[int]$_.ProcessId]}})
foreach($p in $Owned){
  if($p.Name -notmatch '^(powershell|python|sumo|conhost)\.exe$'){throw 'Unexpected process in owned tree'}
  if($p.Name -ne 'conhost.exe' -and $p.ProcessId -ne 58280 -and $p.CommandLine -notmatch 'recover_prospective_20261027\.py|papper\\\.venv\\Lib\\site-packages\\sumo\\bin\\sumo\.exe'){throw 'Unexpected child command'}
}
$Handles=@{}
foreach($p in $Owned){
  try{
    $h=[Diagnostics.Process]::GetProcessById($p.ProcessId); $null=$h.Handle
    if([Math]::Abs(($h.StartTime.ToUniversalTime()-([DateTimeOffset]$p.CreationDate).UtcDateTime).TotalMilliseconds) -gt 2){throw 'PID reused'}
    $Handles[[int]$p.ProcessId]=$h
  }catch{throw "Could not retain original child handle: $($p.ProcessId): $_"}
}
[void](New-Item -ItemType Directory -Path $Out)
$Approval=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');human_approval_received=$true;human_reply=(-join ([char[]]@(25209,20934,20572,29992,26087,21069,32622,35797,39564,65292,20248,20808,26032,20840,37327,39564,35777,65288,25512,33616,65289)));question_item_id='call_c228a8a4febc48dfb9f19ac4a1417938:0';proposal_sha256=(Get-FileHash -LiteralPath 'artifacts/proposals/legacy_attempt_withdrawal_20261027/proposal.json' -Algorithm SHA256).Hash.ToLower();objective_changed=$false;legacy_status='withdrawn for resource priority; incomplete; no certificate';old_alpha=0.025;new_alpha=(0.05/6);old_alpha_reassigned=$false;new_sumo_runs=16384;partial_effects_read=$false}
[IO.File]::WriteAllText((Join-Path $Out 'authorization.json'),($Approval|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
$Before=$Owned|Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine
[IO.File]::WriteAllText((Join-Path $Out 'processes_before.json'),($Before|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
$Results=@()
foreach($p in $Owned){
  $h=$Handles[[int]$p.ProcessId]; $requested=$false
  if(-not $h.HasExited){$h.Kill();$requested=$true}
  if(-not $h.WaitForExit(5000)){throw "Owned process did not terminate: $($p.ProcessId)"}
  $Results+=[ordered]@{pid=$p.ProcessId;created=$p.CreationDate;stop_requested=$requested;actual_exit_code=$h.ExitCode;exit_witness='retained original native process handle';not_a_successful_scientific_completion=$true}
  $h.Dispose()
}
$After=@(Get-CimInstance Win32_Process|Where-Object{$Depth.ContainsKey([int]$_.ProcessId)})
if($After.Count -ne 0){throw 'Original process identity remains or PID reused; inspect'}
$Receipt=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');authorization_sha256=(Get-FileHash -LiteralPath (Join-Path $Out 'authorization.json') -Algorithm SHA256).Hash.ToLower();processes=$Results;all_original_tree_processes_absent=$true;legacy_completed=$false;legacy_certificate_obtained=$false;data_deleted=$false;unknown_historical_exits_rewritten=$false;automatic_legacy_continuation_disabled=$true}
[IO.File]::WriteAllText((Join-Path $Out 'process_stop.json'),($Receipt|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
$Receipt|ConvertTo-Json -Depth 8 -Compress
