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

## Development amendment E: accelerated arrival representation

D completed all 2,097,152 new PPO transitions, 2,560 full development executions
and the on-policy audit. It improves team mean relative to strong ego planning
by 1.1371%, but ramp cost is still .1663 seconds worse. Compared on the SAME D
cases, D's ramp cost is .0121 seconds worse than C's. Therefore the curriculum
did not solve the single-agent prerequisite and is not promoted to joint MAPPO.

A bounded post-hoc diagnostic selected 8 best/8 worst D episodes, not a new
efficacy set. All first command differences occurred at step zero. Several
selected failures have actor max probabilities above .99999 and nominal
one-step-then-baseline suffix regret above 6 seconds. These are illustrative
selected cases, not a prevalence estimate. Softmax gating is thus unsupported.

E appends 10 deterministic local features: own accelerating arrival times at
conflict entry/exit and neighbors' relative entry/exit/goal margins, using the
analytic constant acceleration 2.5m/s^2 and speed cap 20m/s. These are feature
approximations, not predictions with a guarantee or future-state/oracle inputs.
They address the incorrect ordering that a constant-speed ETA can imply when
cars accelerate. Actor inputs are 36 features; central critic adds all public
feature functions to the 13 global state features. The PPO algorithm is unchanged.

Reuse the ENTIRE previously audited 8,192-label BC TRAINING dataset (no new
expert trajectory or held-out label), initialize random weights with seed 4505
and train the new actor for 400 supervised epochs. Then collect 2,097,152 new
on-policy PPO transitions from ORIGINAL starts, no curriculum; lambda=1,
entropy=.005, actor LR=3e-5, critic LR=3e-4. The final model, its BC-only
initializer and parent D must all be compared against both unchanged strong
planners on fresh development initials 94900001--94900128, four modes.
The extra BC fitting, representation and PPO settings are a multi-factor
development attempt; no isolated causal effect or independent replication is
claimed. Passing imitation accuracy alone cannot pass the prerequisite.

## Development amendment F: separate startup credit and cost-sensitive initialization

E completed the fixed 2,097,152 PPO transitions and 3,072 development executions.
It improves ramp cost by .1729 seconds relative to BC alone but still loses
to the unchanged strong ego planner by .1804 seconds. No joint expansion.
Training-label diagnostics show only 431 initial decisions among 8,192 labels:
initial accuracy .6613 versus .9376 after the first decision. These are training
diagnostics, not a held-out classifier score or a guarantee about physical harm.

F uses a local-clock-routed actor with two MLP heads, one for t=0 and one for
t>0. The only routing input is the actor's already public local clock. Both
heads remain shared across vehicles and train with the same PPO surrogate;
the centralized critic is unchanged. This is an architecture/initialization
adaptation of MAPPO, not a new policy-gradient algorithm. F has more actor
parameters, which must be disclosed rather than called capacity matched to E.

Initialize both heads from completed E's PPO actor. Freeze the flow head while
fitting the startup head on exactly 8,192 NEW training initials, seed base
951000001, alternating nominal/dense. For each of nine admissible initial
proposals evaluate the same fixed nominal one-step-plus-rule-tail objective
used by the strong ego planner, recording every counterfactual transition.
The target distribution is softmax(-cost/.25 seconds), which does not punish
physically equivalent near-tied actions as arbitrarily different hard labels.
Fit startup head for 400 epochs, LR 3e-4. This is COST-SENSITIVE EXPERT
INITIALIZATION, not PPO and not a reuse of PI weights or evaluation data.

Seed 4506 then collects 1,048,576 NEW on-policy PPO transitions from ORIGINAL
starts. Flow-head LR=3e-5, startup-head LR=3e-4, critic LR=3e-4, lambda=1,
entropy=.005. PPO updates never read expert costs/targets. Save final weights
and independently audit the log probabilities, GAE and gradients of both heads.
Evaluate F, its separate initializer and parent E against both unchanged strong
planners on 128 new development initials 95000001--95000128, four modes. Keep
all methods/cases. The target remains positive ramp improvement over the strong
ego planner before formal joint learning, with actual-SUMO verification and
separate final confirmation required. No guarantee of achieving this is made.

## Development amendment G: startup approximation quality

F completed 1,048,576 real PPO transitions and the full 3,072-execution
development comparison. Both actor heads have verified nonzero PPO gradients.
Ramp cost still exceeds strong ego planning by .1434 seconds; the prerequisite
remains unmet. Its initializer's in-sample greedy nominal startup regret is
.4990 seconds, with 871/8,192 training starts above one second. This motivates
improving the supervised startup approximation rather than increasing PPO steps
without a diagnosed cause.

G preserves F's completed PPO flow head exactly. The startup head is a fresh
ReLU256 network; flow and centralized critic remain Tanh128. Reuse the ENTIRE
audited F expert cost table, no additional expert query or held-out data. Fit
400 epochs with LR3e-4 to expected expert regret plus .025 seconds times the
soft-target cross entropy. Invalid action costs are masked before multiplying
probabilities, avoiding 0*infinity. This is a separately labeled model-based
initializer; it is not MAPPO or a new policy-iteration algorithm.

Seed 4507 performs 1,048,576 NEW on-policy PPO transitions from ORIGINAL starts,
both actor-head LRs3e-5, critic LR3e-4, lambda1, entropy .005. PPO updates do
not read expert costs. Evaluate final G, parent F and its BC-only initializer
against both unchanged strong planners on fresh 128 development initials
95100001--95100128, four modes. Full source, all failures and final checkpoints
are retained; no intermediate model selection. Architecture and initialization
differences must be disclosed. No success is inferred from training regret.

## Conditional development H: non-saturating expert-logit regression

G's completed initializer has training greedy nominal regret .5244 seconds
(1,014 starts above one second), before its fixed PPO run/test outcomes are
known. Thus merely increasing startup capacity and minimizing probability-
weighted expert cost did not fit the fixed table better. A probability-weighted
loss can have very small gradients for disfavored actions. Prepare H's separate
initializer concurrently using the same full audited table and SAME fixed F
flow parent, with seed4508. No G test outcome is used to prepare it.

H uses ReLU256 and 400 supervised epochs/LR3e-4, but regresses RAW startup
logits to -(candidate cost - minimum candidate cost)/.5 seconds, masked MSE.
Every admissible action has a regression gradient independent of its current
softmax probability. This is a fixed-expert model-based initializer, not
on-policy learning or policy iteration; all expert queries are reused, not new.
It does not ensure low actual cost or generalization.

ONLY if G's completed original-start comparison still fails the single-agent
prerequisite, H performs 1,048,576 NEW on-policy PPO transitions, seed4508, both
actor-head LRs3e-5, critic LR3e-4, lambda1, entropy.005, original starts. Compare
final H, parent F and its initializer against both unchanged strong planners on
fresh development seeds 95200001--95200128, four modes, retaining all outcomes.
G always completes its original registered budget and primary comparison.

## Independent development I: explicitly disclosed legacy-policy warm start

The previous no-PI-initialization restriction was a DEVELOPMENT design choice,
not a user requirement or a definition of MAPPO. It is revised explicitly to
avoid repeatedly learning an already verified single-car skill. The completed
V14 source has three trained single-ego policies, full SUMO/replay audits and
nine passed criteria. These are old MODEL-BASED PI results, never MAPPO results.

Create a separate non-PPO initializer from V14 seed901, checking the original
prerequisite hashes. Copy its ReLU256 ego weights exactly into actor1. Initialize
separate actors0/2 with highest-admissible greedy preferences but non-saturated
stochastic .4*LEVELS logits. All three actors have separate parameters and local
inputs. The centralized critic is new. Scale sampling logits by five, while
greedy inference uses the raw logits, preserving old deterministic actions.
Verify conversion on 1,024 kinematic fixtures; this is not new performance data.

I runs 262,144 NEW on-policy single-agent PPO transitions, seed4601, actor LR1e-5,
critic LR3e-4, gamma1/lambda1, entropy.001. Only ramp actor loss is active in
this prerequisite; actors0/2 are not claimed trained until joint MAPPO. The PPO
code uses no expert labels, counterfactual targets or policy-iteration updates.
The final PPO model must be compared separately with its UNCHANGED legacy
initializer, strong ego/team planners and rule on 128 fresh development initials
95300001--95300128 and four modes. The old certificate/score is not transferred.

This is PI-WARM-STARTED MAPPO, not from-scratch model-free learning or a sample-
efficiency win. Old training cost is inherited and must be disclosed. Positive
new performance cannot automatically be attributed to PPO; compare against the
initializer and require actual multi-agent behavioral contribution in joint
training. No unchanged legacy actor is renamed a trained MAPPO policy. All
earlier failed attempts and conditional H retain their original full budgets.

For I and later training, reset the proposal/minibatch Torch RNG after model
construction to seed+451000000; the vector-environment RNG remains
seed+450000000. This prevents critic architecture/initializer construction
from consuming different amounts of the subsequent proposal RNG stream.
Already running H retains its earlier frozen source and RNG convention.
