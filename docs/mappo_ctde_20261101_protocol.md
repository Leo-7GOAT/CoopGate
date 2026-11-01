# MAPPO replacement: development protocol, 2026-11-01

Status: implementation/development, not a new performance or risk certificate.
The completed V16--V44 studies and their failures remain unchanged. Their
policy-iteration checkpoints are NOT MAPPO weights, and their certificates
cannot be transferred to this learner.

## Algorithm identity

Implement feed-forward MAPPO following Yu et al. (NeurIPS 2022), with a
parameter-shared local categorical actor, a centralized team-value critic,
joint synchronous on-policy trajectories, finite-horizon GAE, per-agent clipped
PPO ratios, entropy regularization, clipped value regression and gradient
clipping. IPPO replaces only the critic information/aggregation. Actors see
18 ego-centric observed features (including visible neighbor command history),
never the centralized state, hidden disturbance mode, seed, future errors or
critic output. The physical command filter remains centralized.

PPO log probabilities refer to sampled pre-filter action indices, NOT executed
commands. There are no planner action labels, counterfactual policy iteration,
distillation, replay-buffer updates or planning-based reward labels. Actor and
critic start from random weights; the joint stage may explicitly initialize
from the new single-agent PPO checkpoint, whose hash must be reported. No old
policy-iteration initialization is permitted.

The 50-decision, 25-second task is a genuine finite horizon. Arrival and horizon
are terminal boundaries (no bootstrap); collection cutoffs bootstrap but do not
reset an episode. A common team return is used; inactive vehicle proposals are
excluded from actor/entropy losses. A fixed 10-second return scale is used for
value regression, not a change to the evaluation objective.

## Bounded development before confirmation

1. Smoke tests and numerical/gradient audit, then a single-ramp-agent PPO run
   of 262,144 joint transitions, seed 4501. Other vehicles propose their
   highest slew-admissible acceleration. This is a single-agent prerequisite,
   not evidence of multi-agent collaboration.
2. A joint MAPPO development run of 1,048,576 joint transitions, seed 4511,
   explicitly initialized from the preceding PPO model. All three actors'
   local proposals now contribute to shared-parameter updates.
3. Fresh development initial seeds 94500001--94500128, equal nominal/dense
   quotas and four original disturbance modes. Compare the strong ego planner,
   strong cooperative planner, highest-admissible rule, single PPO and joint
   MAPPO. First use the exact vectorized physical simulator; SUMO verification
   must independently match trajectories before any traffic result is accepted.
   These are development data, not untouched confirmation data.
4. Final-budget checkpoints are primary. Intermediate checkpoints may diagnose
   learning curves but cannot silently replace the primary endpoint. Any
   subsequent configuration gets a new directory and prospective amendment;
   all failures and time spent remain visible.

Training uses the existing exact longitudinal simulator and common robust
physical filter, with four disturbance modes bounded by 0.5. It is model-free
PPO learning in a simulator, not training on live SUMO invocations. This first
stage matches V15 physics; actuator-memory deployment and its existing strong
recovery baseline require a separate adapter and independent validation.

Three independent training seeds, IPPO, single-agent substitution, actuator
classes, adoption-matched gate comparisons and per-selector actual runtime are
later acceptance stages. Their plans must be fixed before their held-out
outcomes are inspected. No claim of MAPPO beating the strong baseline is made
until the paired results exist; negative results do not authorize weakening
the baseline, deleting seeds or renaming the old learner.

Main research question remains the gate: retain useful learned behavior while
controlling individual baseline-relative degradation. MAPPO is the authentic
learning backbone, not a novelty claim by itself. Neither softmax nor an
unvalidated margin is a per-state safety probability. Prior finite-bank/risk
results apply only to the previously frozen full controller.

Reference: https://arxiv.org/abs/2103.01955 and
https://github.com/marlbenchmark/on-policy (method guidance; this is an
independent traffic adaptation, not an unchanged original-author executable).

## Development amendment B (before B training or evaluation)

Run A completed 262,144 steps and its full 128-initial/4-mode development
comparison. It underperformed the strong ego planner by 1.182 seconds in mean
ramp cost. Do not launch the joint stage from this unsuccessful prerequisite.
The original run, on-policy audit and every comparison remain retained.

One bounded follow-up B, seed 4502, trains single-agent PPO from scratch for
2,097,152 joint transitions, entropy coefficient .02. Eight known kinematic
functions (relative ETA, time headway, near-neighbor and ordering indicators)
are appended to the SAME local observations; the global critic receives their
concatenation in addition to its existing public state. No new hidden
information, teacher labels, planning reward or action imitation is introduced.
This changes representation, training budget and entropy together, so cannot
isolate their separate causal effects. Evaluate the fixed final model on fresh
128 development initials, seeds 94600001--94600128, both regimes/four modes.
All original strong baselines remain. This is development, with no confirmatory
p-value or risk certificate. A further attempt requires another explicit
prospective amendment and preservation of B regardless of its outcome.

## Development amendment C: separately declared expert initialization

B completed all 2,097,152 transitions and 2,048 paired development executions.
It still lost to strong ego planning: ramp cost 10.5605 versus 10.0291 seconds.
No joint training is authorized by these negative development results.

To avoid indefinitely repeating random-initialization runs, C explicitly relaxes
the earlier DEVELOPMENT choice of no teacher initialization. This changes the
training method, not the learning-superiority target or the strong baseline.
The user requested authentic MAPPO, which permits a disclosed supervised
initializer followed by actual on-policy PPO. This is NOT called from-scratch
MAPPO, and imitation alone cannot establish RL benefit.

Generate exactly 8,192 fresh strong-ego-planner proposal labels starting at seed
948000001, alternating nominal/dense and cycling four disturbance modes. Train
the local actor for exactly 80 supervised epochs, smoothing .05, LR 3e-4; save
this BC-only actor separately with a non-PPO algorithm identity. No old PI
weights or PI labels are used. No held-out actions/outcomes are teacher data.

Seed 4503 then performs 2,097,152 new on-policy single-agent PPO transitions
with the same kinematic architecture, entropy .02, GAE lambda=1 (Monte Carlo
credit across episode boundaries, with normal rollout-cutoff bootstrapping).
The PPO collector/update code never reads the teacher dataset. Compare the
fixed final PPO actor AND BC-only initializer to both strong planners and rule
on 128 new development initials, seeds 94700001--94700128, four modes. Also
report paired PPO-minus-BC improvement, not just PPO-minus-planner. Preserve
all initializers, updates and outcomes. This multi-factor development attempt
cannot isolate the causal benefit of any one hyperparameter.

## Development amendment D: unlabeled reset-state curriculum

C's complete 128-initial/four-mode development comparison has 2,560 executions
and zero failures. PPO improved ramp cost versus its BC-only initializer by
0.7859 seconds (development one-sided t lower .5548), demonstrating real
policy-gradient benefit. It still lost to strong ego planning in ramp cost by
.2676 seconds. The small positive mean team gain .4211% has a negative t lower;
there is no passed single-agent prerequisite and no joint training yet.

D retains C's actor and critic, with an explicitly fresh optimizer and seed
4504. Collect 8,192 UNLABELED recoverable states from fresh highest-admissible
rule trajectories starting at seed 950000001. Include actual states with ramp
position in [-65,-15], carrying exact float64 positions, speeds, previous
commands and global clock. No old held-out states, expert actions, returns or
PI labels enter this pool. Every source rollout must remain physically valid;
no rejection of unsafe source trajectories is allowed.

Half of resets use original starts; half uniformly sample this fixed pool and
draw fresh future disturbances. Actual PPO trajectories are newly collected
on-policy from that initial-state mixture; stored actions/returns are never
replayed as PPO samples. Complete 2,097,152 new joint simulator transitions,
same kinematic architecture, entropy .02 and lambda=1. Training metrics on
curriculum episodes are suffix costs, not original-start performance.

Evaluate the fixed final actor from the unchanged ORIGINAL start sampler on
fresh 128 development initials 94800001--94800128 and four modes. Compare D,
its parent C, both strong planners and highest-admissible rule. Preserve all
methods, cases and failures. State-reset training changes the training
objective's initial-state law, and does not guarantee original-start gains.
