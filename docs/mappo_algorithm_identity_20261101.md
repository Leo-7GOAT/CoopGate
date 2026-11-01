# What the new MAPPO implementation actually optimizes

This note explains implemented mathematics, not a new theorem or a superiority
claim. The old policy-iteration models retain their original identity. New
joint checkpoints use `mappo_ctde_20261101` and pass independent rollout audits.

## Three policies and one centralized value function

For roles i=0,1,2, the joint proposal distribution factorizes as

\[
 \pi_\theta(z_t\mid o_t)=\prod_{i=0}^2
 \pi_{\theta_i}(z_{t,i}\mid o_{t,i},M_{t,i}).
\]

J/K/L have three independent two-hidden-layer ReLU actors of width256. Each
consumes18 local observed features and8 deterministic kinematic transforms of
those SAME observations. Locality does not mean perfect full-state sensing:
neighbor entries are range masked. The categorical proposal has13 levels,
-5 through2.5 m/s² in0.625 increments. The per-car mask imposes the command
slew limit. L's positive sampling-logit multiplier5 leaves greedy argmax
unchanged. It does not turn the softmax maximum into safety confidence.

The critic consumes13 public global-state features plus the24 redundant
kinematic transforms for all roles. Its scalar team value is expanded across
roles for the common-payoff loss. It is not called by deployment `proposal`.
The correlated disturbance mode is hidden; these critic inputs are not
claimed to form a complete Markov state of the disturbance generator.

## The physical filter is part of the environment/controller

Let a_t=levels(z_t). Training applies the fixed joint physical filter
u_t=S(x_t,u_(t-1),a_t) before the simulator evolves. PPO stores the likelihood
of **sampled z_t**, not the likelihood of filtered u_t. Different proposals
can produce the same executed command; assigning their likelihoods to u_t
would be wrong. A fixed, potentially nondifferentiable S can be included in
the transition kernel, so a score-function policy gradient does not need to
differentiate through S. PPO's clipped surrogate remains an approximation;
this observation does not establish monotonic policy improvement.

Only proposal generation is decentralized. The physical filter and subsequent
actuator recovery gate are centralized. No fully distributed controller,
general-sum Nash equilibrium or real-car experiment is claimed.

## Common objective, shaping and PPO updates

With I_(t,i) indicating that vehicle i has not reached its goal, stage cost is

\[
 c_{t,i}=I_{t,i}\left(0.5+0.05|u_{t,i}-u_{t-1,i}|\right),\qquad
 c_t=\frac13\sum_i c_{t,i}.
\]

The common reward is -c_t+Phi_(t+1)-Phi_t. Phi is minus mean remaining distance
divided by the speed cap; it is set to zero at every terminal boundary.
For gamma=1 its sum telescopes to a policy-independent initial offset. The
finite horizon is25s/50 decisions. An incomplete vehicle retains its capped
time/cost; completion and physical failures are also reported separately.
Mean team cost does not itself protect each vehicle from performance harm.

For the centralized critic V, collected data give

\[
 \delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),\qquad
 \hat A_t=\delta_t+\gamma\lambda(1-b_t)\hat A_{t+1}.
\]

Here d masks terminal bootstrapping and b stops propagation across resets.
The current finite-horizon task makes time-limit boundaries terminal. A
rollout cutoff alone is neither terminal nor reset. The collector records the
pre-reset following observation for bootstrap, then samples its next action
from the actual reset environment input. J/K/L use gamma=lambda=1.

For each active role, rho_(t,i)=pi_theta_i(z_(t,i)|o_(t,i))/
pi_old_i(z_(t,i)|o_(t,i)). The actor maximizes the mean of

\[
 \min\{\rho_{t,i}\hat A_t,
       \operatorname{clip}(\rho_{t,i},1-\epsilon,1+\epsilon)\hat A_t\}
 +\beta\mathcal H(\pi_{\theta_i}).
\]

Updates use only newly collected joint rollouts, with clipped value regression,
entropy and gradient clipping. No expert label, planner target or PI update
enters this PPO loss. An active-role mask excludes already-cleared vehicles;
their proposals are overwritten by the environment and cannot influence the
executed action. All three roles are trainable in the joint stage. L uses
actor learning rates[1e-4,1e-5,1e-4], not a frozen ramp actor.

## Initialization and evidence limits

I copies an explicitly audited old PI ramp actor, then performs262,144 NEW
single-agent PPO transitions. Its measured performance versus the strong ego
planner is useful, but it has not shown an additional ramp gain over that
unchanged initializer. J/K/L copy I, reinitialize the centralized critic and
perform actual joint PPO updates. K/L additionally use a disclosed broad
rear-coasting initial prior. None is a from-scratch sample-efficiency result.

An audit checks source/weight hashes, independent GAE, first/final rollout
likelihood reconstruction, each role's gradient and parameter changes, actual
sample counts and fixed final budget. These establish algorithm provenance.
Behavior-level cooperation additionally needs paired role replacement and
strong-planner comparisons. Nonzero gradients alone do not establish useful
coordination; J's rear-role substitution has exactly zero outcome effect.

## Why a new gate study is still required

The actuator deployment controller applies a different complete mapping G
from local actor proposals, observed state and executed-command history to
the actual command. Neither the single-PPO performance result nor any earlier
PI risk certificate transfers automatically to G with new actor weights.

Physical stopping certificates require their declared class bounds and valid
memory intervals. Finite-bank performance margins cover only the fixed
prediction paths and reference dynamics; they are not calibrated probabilities
or guarantees of episode-level improvement. Repeated locally favorable
switches do not by themselves prove a globally favorable closed-loop episode.

The remaining study must evaluate the entire fixed gate on fresh independent
initials, retain conditional harmful outcomes among actually adopted episodes,
and measure adoption and utility jointly. Adoption-matched alternatives are
necessary to separate abstention from a better risk discriminator. Actual
whole-callback runtime must be charged to each method independently. These
are the substantive remaining research questions, not solved by renaming a
policy or adding a threshold to softmax confidence.
