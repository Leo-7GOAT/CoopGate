# Fixed mainline queue, 2026-10-26

`advance_mainline_20261026.ps1` removes repeated manual/model polling and gaps
between the already authorized studies. It schedules existing frozen commands;
it does not implement a new experiment or revise any scientific condition.

It first attaches retained Windows native handles to the currently running
calibration and OOD watchers, checking launch receipts and creation times. OOD
analysis starts after its actual zero exit. Calibration analysis waits for both
existing watchers; the original analysis then independently verifies the full
scope and audit receipts. Original confirmation, V42 binding and both stages,
V44, candidate factorial, and the CSPI supplement follow through their existing
entry points. Each command must return zero; the original scientific pass/fail
decision is preserved, and an unsuccessful scientific result is not retried.

No existing execution directory can be restarted. The queue itself refuses an
existing journal. An error preserves every prefix and records the reason; it
does not kill, replace, resume, or weaken a study. Existing machine-quiet,
source-hash, committed-input and full-replay checks remain inside the original
commands. Raw databases stay local and are never staged automatically. Only
explicit completed receipt/analysis directories and the fixed CSPI export are
committed. Completed receipt delivery is pushed to the authorized public repo.
Raw archive delivery and scientific/paper review remain explicit final work.

Operational state and command exits are written only on transitions to
`.git/mainline-20261026.jsonl`. Study progress stays at the original paths. The
queue opens hidden Windows child processes and does not shut down the machine.

Validation: `powershell.exe -NoProfile -File operations/test_mainline_queue.ps1`
checks real native zero/nonzero exits, creation/PID mismatch rejection,
command-chain failure, existing execution rejection and journal rejection.
These checks do not start SUMO or read any unfinished study's effects.
