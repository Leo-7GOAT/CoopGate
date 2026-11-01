# MAPPO mathematical contract and limits

## Proposal-space learning with a shared physical filter

Let the joint sampled categorical proposal be a=(a_1,a_2,a_3), each a_i drawn
from pi_theta(.|o_i). The simulator executes u=S(z,a), with deterministic
physical filtering S, and returns the common reward r. The learned proposal
policy is decentralized; S and the critic need not be. S can map many proposals
to the same physical command. The probability of u is therefore generally not
pi_theta(u|o), and assigning the executed command's index to a sampled action
would be an invalid likelihood calculation.

The actual sampling law is

    p_theta(a|o) = product_i pi_theta(a_i|o_i).

For a fixed filter/simulator, the likelihood-ratio gradient of the trajectory
return is expressed using sums of log pi_theta(a_i|o_i). There is no need to
differentiate the safety filter. This is the ordinary stochastic-policy
gradient argument, not a new theorem. The saved pre-filter proposals and their
log probabilities are the quantities in PPO. Inactive vehicles, whose outputs
are overwritten by the filter after arrival, are excluded from actor loss.

MAPPO uses per-agent clipped ratios and a centralized value estimate. This
surrogate is not an exact optimizer of the joint-policy improvement objective;
it does not imply monotonic policy improvement or a Nash equilibrium.

## Finite horizon and reward

For vehicle i, stage cost while active is

    c_i,t = 1{x_i,t < goal} [dt_control + .05 |u_i,t-u_i,t-1|].

The joint team objective is the mean accumulated vehicle cost. The single-agent
prerequisite instead uses the ramp cost. Training adds a potential difference
Phi(z_next)-Phi(z), with gamma=1 and terminal Phi=0. The finite sum telescopes
to -J-Phi(z_initial); the extra term is independent of the learned policy.
This standard shaping identity does not establish a new safety guarantee.

The decision clock is part of critic state. The 25-second horizon is a true
terminal task boundary. Its value bootstrap is zero. Stopping a rollout before
the horizon is a collection boundary, not an environment termination; its next
value is used. Completed episodes reset before the next action is sampled.

## Advantages and partial observation

TD residuals and GAE are

    delta_t = r_t + gamma (1-terminal_t) V(z_next) - V(z_t)
    A_t = delta_t + gamma lambda (1-boundary_t) A_next.

The critic uses all current public kinematics and executed command memory;
the actor uses only its local observation and deterministic local feature
functions. Neither receives the randomized disturbance mode or future tape.
Thus this is a partially observed stochastic cooperative game. The current
observation is not asserted to be a sufficient Markov state of the hidden
disturbance generator, and approximate GAE is not claimed to be an unbiased
true-state advantage. Optional ETA/headway features do not add information.

## Separation from deployment gate certification

MAPPO maximizes an expected training objective. It does not ensure that each
vehicle improves relative to the strong baseline in every scenario. The gate
has to assess the complete deployed combination of learned proposals, physical
filter, traditional fallback and switching rule. Changing policy weights changes
this closed-loop controller. All earlier finite-sample risk certificates remain
attached to their old frozen controllers; new models require independent
calibration/validation. Actor entropy is not a calibrated safety probability.

Reference algorithm: Yu et al., The Surprising Effectiveness of PPO in
Cooperative Multi-Agent Games, NeurIPS 2022, https://arxiv.org/abs/2103.01955.
