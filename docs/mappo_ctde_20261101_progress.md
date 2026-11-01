# MAPPO replacement: verified progress

## Current authoritative milestone (read this before historical notes)

- I (`single_4601`) is genuine newly updated single-PPO with an explicitly
  audited V14 PI initialization. All 2,560 actual SUMO executions and all
  63,227 commands / 316,135 physical substeps independently replayed. Ramp
  cost gain2.886%, time2.339%, variation17.708%; original numeric thresholds
  and development cluster intervals are met. There is only ONE new PPO seed,
  not the full original three-seed acceptance. Ramp cost versus the untouched
  initializer is .0242s worse, so additional PPO ramp benefit is unproved.
- J (`joint_4701`) completed 1,048,576 joint transitions, all three independent
  actors' parameter/gradient audits, 4,608 exact comparisons and 72 actual SUMO
  integration/replay runs. Mean team gain vs untouched initialization .4218s,
  but raw team cost is 1.2213% worse than strong team planning. Ramp cost
  deteriorates by1.3038s; restoring rear actor2 has zero effect on all512
  trajectories. This is authentic MAPPO execution with inadequate cooperation,
  not a passed final efficacy result. Its planned large-SUMO follow-up was not
  enabled by the failed development condition; no result or sample was deleted.
- K (`joint_4702`) completed 2,097,152 genuine joint transitions, all three
  actor update audits, 4,608 exact comparisons and 72 actual SUMO runs.
  All 1,882 commands / 9,410 physical substeps independently matched. Its
  team cost is .4050% worse than strong team planning; ramp cost is1.1611s
  worse. It has zero raw final failures, but does NOT pass superiority.
  Full SUMO follow-up was not enabled by its prospective condition. Restoring
  the always-coasting rear initializer can cause failed arrival, so that
  comparison alone cannot establish useful conditional rear cooperation.
- L (`joint_4703`) completed all native commands with actual exit0. Fixed
  budget2,097,152, parent `single_4601`, same disclosed rear-coasting prior.
  Ego actor LR1e-5, other actors1e-4: all three receive on-policy updates.
  Adds a stronger role2 replacement with the unchanged highest-admissible
  parent actor, in addition to the three original-role replacements.
  Fresh128 initials / four modes / ten methods =5,120 exact executions;
  80 actual SUMO integration runs and replay. Raw final failures0. Team
  mean gain vs strong TEAM is-.00417s, t-lower-.06457s; the mean paired
  percentage gain is-.17288%. Superiority is NOT established and full3,072
  SUMO was not enabled by the prospectively registered condition.
  Restoring front actor0 loses .26143s mean team cost (development t-lower
  .19433s). Restoring rear actor2 to the highest-admissible parent loses
  .03031s (t-lower .01855s). Both non-ramp actors now have actual behavioral
  contribution; this remains one development seed, not a new risk certificate.
  See `docs/mappo_joint_ego_stable_20261101_protocol.md`.
- The new audited MAPPO deployment adapter connects actual joint actor weights
  to the existing actuator recovery/finite-bank gates. Physical baseline parity
  and modified-weight rejection are tested. It issues NO inherited or new risk
  certificate; actuator/MAPPO performance, matched adoption, latency and fresh
  risk/adoption/utility tests are still pending.
- The K actor also completed the new recovery wiring smoke: five actual SUMO
  runs, all five selectors, collection/review native exit0, all115 commands
  and575 physical substeps independently matched, including complete memoized
  gate replay. This ONE initial is not efficacy evidence. The finite-bank
  selector adopted no learned action, so its conditional-harm denominator is
  zero/undefined. Training ran concurrently; these callback timings are not
  isolated runtime evidence. The full400-run integration protocol is registered
  separately in `docs/mappo_recovery_integration_20261101_protocol.md`.
- The400-run L actuator recovery/gate integration is COMPLETE: native
  collection session74489 and review77393 both exited0. Every9,332 commands
  and46,660 physical substeps independently matched; complete gate replay
  matched too. Five fresh initial draws underlie the80 conditions per method.
  All methods complete80/80. Bank gate adopts29/80, observes0/29 harmful,
  aggregate cost gain .48458%. Nominal-only gate adopts55/80 with6 harmful,
  gain1.62852%. These are repeated descriptive conditions, NOT independent
  binomial trials or risk-certified results. Bank utility also falls below
  the unchanged .5% formal threshold. Bank timing p95=181.44ms, max1828.96ms,
  41/1840 callbacks>500ms; all costs belong to the actual selector. Runtime
  is explicitly non-isolated (exact evaluation/Git overlap), not hard-real-time
  evidence. No communication-latency measurement is used.
- Two fixed single-PPO replications are ACTIVE: seed4602/session3154 and
  seed4603/session47895. Same I initializer, architecture, hyperparameters and
  262,144-step budget; new fresh95800001/95900001 sets. See
  `docs/mappo_replication_20261101_protocol.md`. Each queue owns its native
  children and conditionally performs full SUMO only if unchanged single-car
  thresholds pass. Do not edit their frozen source/protocol or restart them.
  These are not yet passed, and J/K/L are not three identical-config seeds.
- All completed A--L raw data are retained in their separate byte-verified
  deliveries. E/F/G/H remain negative development outcomes, not discarded
  runs. Old PI studies, their failures and original paper remain unchanged.

L's native pipeline is terminal, not waiting for more training or raw SUMO.
Its development condition failed. Read its completed report/role reverts;
avoid inferring success from training
curves, parameter changes, native zero exits, or a positive initial prior.
The active goal is NOT complete or blocked.

The new goal is active. The former research-delivery goal remains completed;
its experiments, negative results and manuscript are not relabeled as MAPPO.

## Implemented

- Shared decentralized categorical actor, centralized team critic, joint
  synchronous on-policy collection, GAE, clipped PPO/value losses, entropy and
  gradient clipping. PPO updates never read planner action labels and perform
  no policy-iteration updates. A separate, disclosed BC initializer is used in C.
- Single-agent prerequisite and joint-agent training stages with explicit
  active-agent loss masks. Earlier random and BC starts failed. The disclosed
  I amendment uses PI initialization; newly PPO-updated single weights are
  the only warm-start source for the joint stage.
- Snapshot the first/final pre-update rollout, sampling policy, original
  action log probabilities, rewards, terminal flags and complete source hashes.
- Audit reconstructs GAE independently and checks sampled-proposal log
  probabilities, critic and actor gradients, and role-specific gradient paths.
- Complete development comparison retains both strong ego and cooperative
  planners, all disturbance modes, all failures and fixed final checkpoints.

## Completed development A (seed 4501)

262,144 single-agent PPO transitions completed with 9,247 simulated episodes.
The first and final rollout audits passed, log-probability reconstruction error
was zero, and actor and critic gradients were nonzero. This establishes the
PPO implementation's provenance, not its traffic performance.

128 independent development initials, four repeated modes and four methods
produced 2,048 exact-simulator executions. All completed. Single PPO ramp cost
was 11.3662 seconds versus strong ego planning 10.1842 seconds; it failed the
learning-superiority prerequisite. Joint training has NOT started from it.

Actual SUMO integration: the first 32-execution run omitted initial-state
instrumentation in the new evaluator, so the independent reviewer rejected it
with KeyError('initial'). Its original results are retained unchanged.
After fixing instrumentation, a separately named 32-execution integration run
matched the exact physical simulator in all 858 commands and 4,290 independently
reconstructed physical substeps. These repeated executions do not add any
independent initials or support learner superiority.

## Completed development B (seed 4502)

Prospectively amended B: new seed 4502, fresh random initialization, deterministic
local kinematic features, entropy .02, fixed 2,097,152-transition single-agent
PPO budget completed in 303 seconds. Both saved rollout audits passed. Fresh
128-initial/four-mode paired evaluation completed all 2,048 executions without
failures. Ramp cost was 10.5605 seconds versus strong ego planning 10.0291
seconds. Team gain was -1.2368% relative to strong ego planning and -7.4455%
relative to strong team planning. Thus B also fails the prerequisite; no joint
training is started. A/B use different fresh development sets, so their scores
are not a controlled estimate of the representation change's causal effect.

Completed A/B raw data, training weights/curves, source snapshots and failed
integration instrumentation are all preserved in a byte-verified archive:
321 files, 87,208,894 original bytes, 28,281,950 compressed bytes.

## Completed development C (expert initialization plus genuine PPO)

A separately labeled, non-PPO expert initializer was trained on exactly 8,192
fresh strong-ego-planner proposals, 431 simulated trajectories and 80 supervised
epochs. Training action accuracy is 91.31%; this is NOT held-out traffic
performance or PPO benefit. No old PI weights/labels are used.

C completed 2,097,152 actual on-policy single-agent PPO transitions with
lambda=1, seed 4503, starting from that actor. Its PPO collector and updates
never read expert labels. The fixed final model must be compared both with
the strong planners and with the BC-only initializer, on fresh development
seeds. This explicit expert-initialized PPO path supersedes the earlier
development-only no-imitation-initialization choice. It is not from-scratch
MAPPO, and imitation benefit cannot be attributed to reinforcement learning.
The full comparison has 2,560 exact-simulator executions on 128 new development
initials and four modes, all completed. PPO ramp cost improved by .7859 seconds
relative to BC alone (development t lower .5548); this establishes genuine
policy-gradient benefit in this development comparison. However ramp cost
still exceeds strong ego planning by .2676 seconds. Team gain .4211% has a
negative t lower (-.0148 seconds). The prerequisite remains unmet. The full
PPO/GAE audit passed. A separately named actual-SUMO integration matched all
1,002 commands and 5,010 independently reconstructed physical substeps across
40 runs. These repetitions add no independent initials.

All 8,192 BC labels were regenerated from the declared 431 planner trajectories
and matched. Completed C data and its initializer are preserved in a verified
archive of 180 files, 62,645,882 original bytes and 22,286,885 compressed bytes.

## Completed development D: unlabeled reset-state curriculum

An unlabeled reset-state pool was generated from 907 fresh rule trajectories:
8,192 exact float64 near-merge states, previous commands and clocks. No expert
actions, returns, old training states or held-out states enter this pool.
Half of training starts now use these states; the rest retain the original
sampler. On-policy PPO continues from C for 2,097,152 new transitions, seed
4504. A separate 8,192-step numerical smoke run completed; it is not an
efficacy result or an independent training seed. Actual original-start paired
evaluation of the fixed final model completed all 2,560 executions without
failure. Team gain versus strong ego planning is 1.1371% (development t lower
.0751 seconds), but ramp cost remains .1663 seconds worse. Compared on these
same cases, D is .0121 seconds worse for the ramp than parent C. Thus this
attempt did not solve the prerequisite. A further 32 actual-SUMO integration
runs matched 720 commands and 3,600 independent physical substeps.

Post-hoc selected-case diagnostics show very confident wrong INITIAL actions:
all 16 selected best/worst episodes first differ at step zero; some failures
have softmax maxima >.99999 and >6 seconds of nominal suffix regret. Realized
future tapes were used only for offline diagnosis, never by actor/gate. This
cannot estimate population prevalence, but it rules out treating max softmax
as a validated safety confidence.

## Completed development E

Add analytic accelerating entry/exit margins derived exclusively from local
observations. Refit a separately labeled BC initializer on the same 8,192
training labels for 400 epochs, then run 2,097,152 genuine PPO transitions,
seed 4505, original starts, smaller actor LR 3e-5 and entropy .005. Final
paired evaluation includes BC-only and parent D, with unchanged strong
baselines and fresh development seeds. This new hypothesis is unverified.

E's separate initializer completed 400 epochs on the reused training data,
training action accuracy 92.31%, with ZERO additional expert trajectories.
Its actual PPO run completed at
`artifacts/mappo_ctde_20261101/development/single_4505` (fixed 2,097,152 steps).
All 3,072 development executions completed. The ramp remains .1804 seconds
worse than strong ego planning, though PPO improves it by .1729 seconds versus
BC alone. Team gain versus strong ego planning is .6211%. This is not a passed
ramp prerequisite. Forty actual SUMO integration runs independently matched 961
commands and 4,805 physical substeps. All E data are archived (182 files,
70,047,596 original bytes; 24,552,604 compressed bytes).

## Historical notes: prospectively amended F

The continuation commands and session references below are historical and
superseded by the authoritative current milestone above. Do not execute them
again. F/G/H completed without satisfying the prerequisite; I/J/K supersede
these operational instructions. The fixed4602/4603 replications are active.

The startup and subsequent actor heads are now routed only by the already
observed local clock. This separates rare initial decisions from abundant
later acceleration decisions; it increases actor capacity and is not claimed
capacity-matched to E. Both heads use the same on-policy PPO surrogate.

A clearly separate cost-sensitive expert initializer fitted its startup head
on 8,192 fresh training initials and all 73,728 admissible proposal costs, using
1,806,756 nominal counterfactual joint decisions. Its flow head exactly copies
E's completed PPO actor. These expert calls are NOT PPO steps or SUMO runs.
Training greedy nominal regret is .4990 seconds, so fitting accuracy cannot
establish original-start superiority. F's actual PPO run, seed 4506, uses a fixed
1,048,576 new on-policy transitions, startup LR3e-4/flow LR3e-5, lambda1,
entropy .005, original starts. Expert costs are never read by PPO updates.

The previous E continuation commands below are historical and already executed.
For F, wait for root unified-exec training session 64915 to exit zero, audit
`development/single_4506`, then evaluate F, parent E and its BC-only control on
fresh seeds 95000001--95000128. The whole expert table reviewer is running in
session 93640; do not restart it if observation times out. Goal remains active.

### Next executable steps for the active goal

After the actual E training process exits zero (root unified-exec session
23740), run the following once, preserving every output directory:

```powershell
& C:/Users/Leo/Desktop/papper/.venv/Scripts/python.exe tools/audit_mappo_ctde.py artifacts/mappo_ctde_20261101/development/single_4505
& C:/Users/Leo/Desktop/papper/.venv/Scripts/python.exe tools/evaluate_mappo_ctde.py --out artifacts/mappo_ctde_20261101/development/single_4505_exact --paths artifacts/mappo_ctde_20261101/development/single_4504/policy.pt artifacts/mappo_ctde_20261101/development/single_4505/policy.pt --n 128 --seed-base 94900001 --backend exact --expert-initial artifacts/mappo_ctde_20261101/development/bc_4505/initializer.pt
```

Wait for each native zero exit and inspect the COMPLETE report. Judge ramp
cost versus strong ego planning separately from team mean and from BC gain.
Do not reinterpret D's 1.1371% team gain as a passed ramp prerequisite. If E
establishes a positive ramp result, independently verify actual SUMO before
formal joint expansion. If not, diagnose the remaining initial-action errors
before proposing another bounded prospective amendment; do not silently extend
E, weaken the baseline or promote a lucky intermediate checkpoint. A local
clock-conditioned actor head is one possible untested follow-up hypothesis,
not an approved success or an implemented feature.

Every completed raw A/B/C/D directory is delivered in its separate byte-verified
archive; E is unfinished and must not be archived as completed. All original
studies/certificates remain unchanged, and the other protected desktop project
must not be accessed or interrupted. The goal is ACTIVE, not complete/blocked.

Three-agent MAPPO, independent seeds and new gate certification remain pending.

## Next gates

1. Establish the new single PPO's paired performance against the strong ego
   baseline. If it fails, retain that result and diagnose the policy before
   expanding to joint training.
2. True joint MAPPO, independent seeds, IPPO and role substitution; verify
   behavior-level cooperation as well as mathematical update identity.
3. Actuator-memory training/deployment with the SAME strong recovery baseline.
4. Fixed-policy gate calibration and fresh tests, adoption-matched controls,
   selector-specific measured runtime; no transfer of old certificates.
5. Update the paper only with completed new results; publish code, data and
   negative outcomes. No claim of venue readiness until these evidence gaps
   are assessed.
