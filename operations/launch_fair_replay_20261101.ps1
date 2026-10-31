$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$name='CoopGate-FairReplayRecovery-20261101-Interactive'
if(Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue){throw 'Owned task exists; inspect instead of restarting'}
$entry=Join-Path $root 'operations/continue_fair_replay_20261101.ps1'
$log=Join-Path $root '.git/mainline-fair-replay-20261101.stdout.log'
$receipt=Join-Path $root '.git/scheduled_fair_replay_launch_20261101.json'
if((Test-Path $log) -or (Test-Path $receipt)){throw 'Preserve existing launch evidence'}
$arguments='-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -Command "& '''+$entry+''' *> '''+$log+'''"'
$action=New-ScheduledTaskAction -Execute 'C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe' -Argument $arguments -WorkingDirectory $root
$user=[Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal=New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$null=Register-ScheduledTask -TaskName $name -InputObject (New-ScheduledTask -Action $action -Principal $principal -Settings $settings)
$proof=@{utc=[DateTime]::UtcNow.ToString('o');task_name=$name;logon='Interactive';run_level='Limited';triggers=0;automatic_retry=$false;starts_requested=1;source_sha256=(Get-FileHash $entry -Algorithm SHA256).Hash.ToLower();recovery_sha256=(Get-FileHash (Join-Path $root 'artifacts/fair_replay_recovery_20261101/recovery.json') -Algorithm SHA256).Hash.ToLower()}
[IO.File]::WriteAllText($receipt,($proof|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
Start-ScheduledTask -TaskName $name
$proof|ConvertTo-Json -Compress
