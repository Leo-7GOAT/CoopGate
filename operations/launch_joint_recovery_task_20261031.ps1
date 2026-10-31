$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if($root -ne 'C:\Users\Leo\Desktop\papper\CoopGate'){throw 'Unexpected project root'}
$name='CoopGate-JointRecovery-20261031'
if(Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue){throw 'Owned one-shot task exists; inspect, never restart implicitly'}
$entry=Join-Path $root 'operations/continue_joint_recovery_20261031.ps1'
$log=Join-Path $root '.git/mainline-joint-recovery-20261031.stdout.log'
$receipt=Join-Path $root '.git/scheduled_joint_recovery_launch_20261031.json'
if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $receipt)){throw 'Owned launch evidence exists'}
if(-not (Test-Path -LiteralPath (Join-Path $root 'artifacts/joint_risk_recovery_20261031/recovery.json'))){throw 'Frozen recovery registration missing'}
$arguments='-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -Command "& '''+$entry+''' *> '''+$log+'''"'
$action=New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $arguments -WorkingDirectory $root
$user=[Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal=New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task=New-ScheduledTask -Action $action -Principal $principal -Settings $settings
$null=Register-ScheduledTask -TaskName $name -InputObject $task
$proof=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');task_name=$name;principal=$user;run_level='Limited';triggers=0;on_demand_one_shot=$true;automatic_retry=$false;source=(Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLower();recovery_plan_sha256=(Get-FileHash -LiteralPath (Join-Path $root 'artifacts/joint_risk_recovery_20261031/recovery.json') -Algorithm SHA256).Hash.ToLower();starts_requested=1}
[IO.File]::WriteAllText($receipt,($proof|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
Start-ScheduledTask -TaskName $name
$proof|ConvertTo-Json -Compress
