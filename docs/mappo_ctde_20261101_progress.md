# MAPPO replacement: verified progress

## Current authoritative milestone (read this before historical notes)

- I (`single_4601`) is genuine newly updated single-PPO with an explicitly
  audited V14 PI initialization. All 2,560 actual SUMO executions and all
  63,227 commands / 316,135 physical substeps independently replayed. Ramp
  cost gain2.886%, time2.339%, variation17.708%; original numeric thresholds
  and development cluster intervals are met. I's comparison has ONE new PPO
  seed (additional replications below), not full three-seed acceptance. Ramp cost versus the untouched
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
- Fixed single-PPO replication4602/session3154 is COMPLETE, all actual
  native exits0. Full2,560 SUMO executions independently match63,006 commands
  and315,030 physical substeps. SUMO cost/time/variation gains
  3.4089%/2.7576%/20.8277%, all numeric/development interval checks pass.
  This is a second successful seed; the complete three-seed criterion still fails.
  Replication4603/session47895 is terminal with native exit0, but scientific
  competence fails: gains1.5686%/1.0116%/17.5217%, cost/time bootstrap lower
  bounds-.02143s/-.0625s. Full SUMO was not enabled by the registered condition.
  This is insufficient precision at128 initials, not evidence of superiority.
  Its failed outcome and full raw training/evaluation archive are preserved.
  Both use the same I initializer, architecture, hyperparameters and
  262,144-step budget; new fresh95800001/95900001 sets. See
  `docs/mappo_replication_20261101_protocol.md`. Each queue owns its native
  children and conditionally performs full SUMO only if unchanged single-car
  thresholds pass. Do not edit their frozen source/protocol or restart them.
  Three-seed acceptance remains unproved, and J/K/L are not three
  identical-configuration replications.
- Both exact-L JOINT replications are COMPLETE, native pipelines11127/85960
  each observed exit0. Same fixed I parent; each2,097,152 new joint steps,
  5,120 exact comparisons and80 SUMO/replay.4704/4705 team gains vs strong
  TEAM are-.05914s/+.05646s, t lower-.12795s/-.00944s. Neither passes the
  positive-lower-bound condition, so neither full3,072-SUMO stage is enabled.
  Front-role benefit repeats; rear-vs-highest-admissible benefit is negative
  in4704 and positive in4705. Keep this inconsistency, not a best-seed claim. See
  `docs/mappo_joint_replications_20261101_protocol.md`. These are independent
  joint optimization RNG seeds conditional on one parent, not independent
  complete single/joint pipelines. Keep all negative outcomes; no best seed.
  All native handles are terminal; do not restart these completed queues.
- Three-model actuator/gate integration is COMPLETE:400 actual SUMO runs per
  fixed model,1,200 total, all28,118 commands/140,590 physical substeps replay.
  Every repeated baseline trajectory matches across models. SAME five initial
  draws, not15 or1200 independent trials. Bank gains .48458%/.64234%/.55362%,
  adoption29/28/30 of80 conditions, observed harmful0/29,0/28,0/30. No new
  risk certificate or all-model utility pass. Both new collection/review pairs
  exited0 (64905/83212 and84481/20805). See `three_model_gate_matrix.json`.
- Optimistic-arrival rejection completed160 source episodes/320 whole-gate
  replays,7,324 callbacks, all commands/admissions/accepted evidence identical.
  Paired total callback cost falls26.0% nominal/15.9% bank; bank max remains
  1425.26ms,40 calls>500ms. Actual bound work and every rejection-prefix proof
  are recorded. This is replay, not extra SUMO or independent efficacy data.
  Native smoke75905/full75179 exited0. Byte-exact source snapshots are archived.
- Extra nominal-certificate caching also passes complete160/320 replay and
  primitive key/capacity tests, but bank max grows to2073.02ms vs1651.26ms in
  its own paired reference,44 vs43 calls>500ms. Do NOT adopt it as a latency
  solution. Preserve its negative engineering outcome (27060/50128 exits0).
- Original minimum performance margins are all zero at4703's175 nominal/47
  bank adopted decisions. A NEW `ScenarioConfidenceGate` uses complete-path
  pass fractions with registered failed-path budgets0/1/3. Budget0 preserves
  original real predictor admission in four-class tests; synthetic tests show
  threshold switching. Eight targeted tests pass. This is NOT a calibrated
  probability/risk certificate. Actual new-rule SUMO wiring smoke is now
  COMPLETE:10 actual runs,220 commands/1100 physical substeps,collection71871
  and separate review both exited0. New independent reviewer reconstructs
  actuator physics/memory certificates AND all33 score comparisons/whole gate.
  Tampered score/budget/missing-path evidence is rejected in targeted tests.
  This one initial is not efficacy or calibration. See
  `docs/scenario_confidence_gate_20261101_protocol.md`. Do not mislabel its
  allowed positive path differences as strict-bank evidence.
- Full score development is COMPLETE:800 actual SUMO executions split into four
  disjoint200-run shards. Native collection sessions24745/83748/38299/54617
  correspond to shards0/1/2/3 under `development/scenario_score_grid_20261102_shardN`.
  All three fixed models,all budgets0/1/3,one shared baseline,five scenes,four
  classes,four modes; SAME five initials,not20/80/800 independent samples.
  Collections and reviews38484/33650/62613/97697 all observed exit0.
  Complete union has18,844 commands/94,220 physical substeps independently
  matched,with no missing/duplicate cases. Summarized by
  `tools/summarize_scenario_score_grid.py`. Registered in
  `docs/scenario_score_development_20261102_protocol.md`. No new risk budget
  is spent and no threshold is certified from this development grid.
- Strict-budget gains are .85952%/.82096%/.59872%; harmful adoptions0/30,
  0/46,3/33. Budget1 gains .88044%/.84841%/.63925%,harm0/31,0/48,2/35.
  Budget3 gains .95691%/.86279%/.74187%,harm1/39,1/52,2/41. These repeated
  FIVE-initial results show a tradeoff,not certified risk or stable superiority.
  All three strict-budget adverse4705 episodes share slow_yield/lag_strong
  initial96902001. The gate's33/33 score does not prevent actual regression.
- A privileged OFFLINE diagnosis re-evaluates four adopted decisions in those
  adverse episodes with the true realized plant/memory and future tape. Three
  reveal a cost/time regression that the original finite bank admitted.
  This is selected-case model-error evidence,not an online comparator or extra
  SUMO trial. Clean-mode evidence particularly motivates actuator-parameter/
  memory uncertainty assessment; no claim isolates causes in noisy modes.
  See `scenario_forecast_error_diagnosis_20261102.json`.
- Fixed IPPO local-value ablation seed4703/native pipeline98901 is COMPLETE,
  all actual native exits0:2,097,152 joint steps,5,632 exact comparisons,88
  actual SUMO integrations and replay. Versus strong team planning it remains
  .05088s worse,so full3,584-SUMO follow-up was not enabled. On identical
  cases MAPPO's team cost is .04671s better than IPPO (development t lower
  .01792s); this is ONE paired seed,not a replicated superiority claim.
  It preserves exact actor initial tensors,config and proposal
  RNG; audit counts228,135 actor parameters,75,777 centralized versus72,961
  shared-local critic parameters. This is not capacity/compute equality and
  does not imply equal compute. All three paired seeds are registered in
  `docs/ippo_matched_20261102_protocol.md`.4704/session58854 and4705/session15585
  are also COMPLETE,all native stages exited0,each5,632 exact/88 SUMO/replay.
  MAPPO-vs-local mean gains across seeds are+.04671/-.02004/+.02722s; lower
  bounds+.01792/-.04735/-.01063s. No consistent all-seed MAPPO superiority.
  All raw outcomes,source snapshots and checkpoints are archived;no replacement.
- Parameter-scenario prototype preserves original33 noise paths and adds8
  declared endpoint paths with separate executed-history actuator memories.
  Five parameter/memory tests and independent checker tamper test pass.
  Offline prefix probe changes3/3 old oracle-adverse commands,preserves the
  one nonadverse command; this probe is not a closed-loop performance result.
- ACTUAL paired regression fixture is COMPLETE:21 SUMO runs,552 commands/
  2760 substeps,collection79371/review90643 observed exit0. Original4705 has
  harmful3/3 and-1.2863% gain; augmented4705 has harmful0/3 and+2.1186%,STILL
  adopted3/3. Other two models preserve+1.9370% gain/adoption3/3/harm0.
  Only ONE reused initial; no generalization or risk certificate. Maximum
  callback for augmented4705 is1289.46ms,so online cost remains unresolved.
- FRESH parameter development is COMPLETE:560 actual runs,seven methods,four
  disjoint140-run shards under `development/parameter_scenario_grid_20261102_shardN`.
  Collection sessions7908/14251/15152/71687 and review27818/73236/76717/77084
  all observed actual exit0. Fresh base
  seed97000001,five scenes/four classes/four modes,SAME five draws across
  rules/models,not560 independent trials. All13,422 commands/67,110 physical
  substeps independently matched;complete union/identity checked by
  `tools/summarize_parameter_grid.py`. Parameter gains1.65809%/1.68685%/1.71066%,
  adopted31/32/32,harmful1/1/1. Original gains1.70942%/1.66845%/1.73540%,
  adopted37/37/38,harmful2/3/1. No risk certification or same-adoption claim.
  Parameter callback p95=621/763/588ms,max3885/4005/3806ms under disclosed
  contention. This cost is unresolved;do not present the gate as real-time.
- Adoption-matched episode thinning implementation/tests are prepared. Coins
  have separate controller streams;actual executed adoption is NOT the coin.
  No probability is calibrated from these five-initial development sets.
  Actual confirmation/control collection remains required. Error-allocation
  inventory scanned330 metadata files,found23 with allocations,skipped0;
  it allocates NO error and does not yet prove the next attempt unused.
- Fast-gate teacher dataset is prepared at `gate_distillation_data_20261102`:
  5,964 decisions,73 online-only features,29 arbitration classes,EXACT command
  reconstruction for all5,964 labels. Only SIX independent initial groups
  underlie these correlated frames. Labels describe teacher choices,including
  its imperfect decisions;they are not ground-truth no-harm labels or PPO
  transitions. No student is trained yet. Split by whole initial across all
  models/modes/plants,never frames;keep original true MAPPO actors and physical
  baseline. Any student needs independent whole-controller certification.
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
these operational instructions. All4601--4603 single stages,4703--4705 joint
stages and three-model gate integrations are terminal; use current next gates.

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

Historical pending note superseded: three-agent MAPPO and fixed joint seeds
are complete. Stable superiority, IPPO and new gate certification remain pending.

## Next gates

1. Parameter development and all IPPO comparisons are done. The grouped
   supervised fast gate FAILED positive generalization and is not promoted.
   Exact scalar prediction-internal planning completes240-trajectory/11,460-
   callback replay and140 fresh actual SUMO executions with identical original
   commands/evidence. Full callback cost falls about55%;500ms overruns remain.
   See `scalar_nominal_recovery_20261102_results.md` and its complete archives.
2. Retain the fresh shared strong-lag counterexample:all three models have one
   harmful adopted episode in20 independent new initials. Offline true-baseline
   advantage decomposition exactly identifies its first erroneous admission.
   A32-cell observation-consistency observer retains true parameters at3657
   decision boundaries,but its27 representative-point probe STILL misses the
   error. It is NOT an implemented repair or new online controller.
3. Implement/test the explicitly registered normalized worst-path team-gain
   margin in `gain_margin_20261102_protocol.md`;zero-floor parity is required.
   All original per-car tests,nominal ranking,physical certificates,fixed three
   actors and failures remain. No offline episode filtering or threshold
   selection on confirmation. New closed-loop evidence is still required.
4. Register fresh calibration/confirmation, fixed thresholds and new familywise
   error allocation for ALL three models; retain risk<=5%,adoption>=20%,
   utility>.5%. Add the calibration-selected adoption-matched control.
5. IPPO and new timing comparisons are complete;consistent all-seed MAPPO-over-
   IPPO superiority is unproved. Retain4603's failed interval and
   all-model raw-superiority/utility gaps; do not chase a replacement seed.
6. Use new exact-pruning implementation only after fresh actual SUMO verification;
   certificate caching is not a proven latency improvement.
7. Update the paper only with completed new results; publish code, data and
   negative outcomes. No claim of venue readiness until these evidence gaps
   are assessed.
