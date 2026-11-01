# Fixed single-PPO replications before further joint acceptance

I/4601 passed the numeric single-car competence thresholds but represents
only one new PPO seed. J/K/L are distinct development hypotheses, not three
replications of one joint configuration. L has behavioral front/rear benefits
but does not beat strong team planning. Do not claim full three-seed success.

Register single seeds4602 and4603 with I's SAME fixed legacy_901 initializer,
262,144 new on-policy transitions, local ReLU256 role actors, kinematic
features, sampling-logit scale5, actor LR1e-5, critic3e-4, entropy.001,
gamma/lambda1, all other settings unchanged. Only the ramp actor is active
in this prerequisite. Neither teacher actions nor PI updates enter PPO.
The old initializer's newly absent `ego_actor_lr` metadata field normalizes
to its unchanged default0; this is schema compatibility, not weight editing.

Seeds4602/4603 use fresh128-initial development sets95800001--95800128 and
95900001--95900128 respectively. Each compares rule, strong ego, strong team,
the unchanged legacy initializer and fixed final PPO on all four modes:
2,560 exact executions per model. Actual on-policy/GAE/role audits are required.
Then assess the original ramp thresholds: cost>=1%, time>=.5%, variation>=10%,
positive development cluster-bootstrap cost/time lower bounds and no failure.

Only if a model meets those exact development thresholds, run its FULL2,560
actual SUMO comparisons on the same complete paired set, followed by independent
physical/controller replay and the same competence calculation. If it fails,
retain its entire outcome and record the full SUMO stage as not enabled. There
are no retries, early stopping, intermediate-checkpoint choices or discarded
training seeds. Conditional repeated SUMO runs add no independent initials.
These are development prerequisites, not formal gate calibration/confirmation.

Each native pipeline owns its child process handles and records actual exits.
The two pipelines may run concurrently to use separate CPU cores; their callback
timing is not an isolated runtime benchmark. Completed code/input hashes remain
fixed until both pipelines terminate. No source from the other desktop project
is accessed. Future joint replications and their matching single parents must
be registered separately after these results; none is silently enabled here.

The new MAPPO recovery/gate integration continues independently and all of its
failure/adoption/utility/runtime outcomes remain required. No earlier risk
certificate or error budget is reused. Research goal remains active.
