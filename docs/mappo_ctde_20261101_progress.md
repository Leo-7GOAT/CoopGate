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

## In progress

Prospectively amended B: new seed 4502, fresh random initialization, deterministic
local kinematic features, entropy .02, fixed 2,097,152-transition single-agent
PPO budget. Final-budget paired development evaluation and audits are pending.
No success or improvement is inferred from its intermediate training curve.

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
