# MAPPO replacement: verified progress

The new goal is active. The former research-delivery goal remains completed;
its experiments, negative results and manuscript are not relabeled as MAPPO.

## Implemented

- Shared decentralized categorical actor, centralized team critic, joint
  synchronous on-policy collection, GAE, clipped PPO/value losses, entropy and
  gradient clipping. PPO updates never read planner action labels and perform
  no policy-iteration updates. A separate, disclosed BC initializer is used in C.
- Single-agent prerequisite and joint-agent training stages with explicit
  active-agent loss masks. Random initialization; new single PPO weights are
  the only permitted warm-start source for the joint stage.
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

## In progress: prospectively amended E

Add analytic accelerating entry/exit margins derived exclusively from local
observations. Refit a separately labeled BC initializer on the same 8,192
training labels for 400 epochs, then run 2,097,152 genuine PPO transitions,
seed 4505, original starts, smaller actor LR 3e-5 and entropy .005. Final
paired evaluation includes BC-only and parent D, with unchanged strong
baselines and fresh development seeds. This new hypothesis is unverified.

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
