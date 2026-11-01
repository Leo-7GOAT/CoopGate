# MAPPO replacement: verified progress

The new goal is active. The former research-delivery goal remains completed;
its experiments, negative results and manuscript are not relabeled as MAPPO.

## Implemented

- Shared decentralized categorical actor, centralized team critic, joint
  synchronous on-policy collection, GAE, clipped PPO/value losses, entropy and
  gradient clipping. No planner action labels or policy-iteration updates.
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

## In progress: prospectively amended C

A separately labeled, non-PPO expert initializer was trained on exactly 8,192
fresh strong-ego-planner proposals, 431 simulated trajectories and 80 supervised
epochs. Training action accuracy is 91.31%; this is NOT held-out traffic
performance or PPO benefit. No old PI weights/labels are used.

C now performs 2,097,152 actual on-policy single-agent PPO transitions with
lambda=1, seed 4503, starting from that actor. Its PPO collector and updates
never read expert labels. The fixed final model must be compared both with
the strong planners and with the BC-only initializer, on fresh development
seeds. This explicit expert-initialized PPO path supersedes the earlier
development-only no-imitation-initialization choice. It is not from-scratch
MAPPO, and imitation benefit cannot be attributed to reinforcement learning.
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
