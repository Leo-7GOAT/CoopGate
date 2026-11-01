# Joint MAPPO development K: escape the observed coordination equilibrium

J completed 1,048,576 joint transitions, full three-role PPO/GAE/parameter audits,
4,608 exact executions, 72 actual SUMO integration runs and full integration
replay. Raw joint mean cost is 1.2213% worse than strong team planning. Ramp
cost is 1.3038s worse while main-road cars improve by .5161/.4822s. Restoring
rear actor2 exactly to its initial parameters changes NONE of the 512 paired
trajectories' costs; restoring actor0 has a small .0200s team effect. These
results support real joint PPO execution but insufficient effective cooperation.
No final full-SUMO superiority claim is made; the planned positive-development
condition for that large follow-up failed.

K uses the same independently parameterized actors, centralized critic, common
team cost and physical simulator/filter. It starts from the SAME new single-PPO
parent `single_4601/policy.pt`; source/ancestry are disclosed. Only actor2's
initial last layer is changed: weights zero, unscaled bias=-.08*abs(LEVELS).
With sampling scale5 this gives a broad zero-acceleration preference, rather
than the previous highest-acceleration preference. This is an INITIAL PRIOR,
not a fixed rear-car rule. All three actors still sample and learn jointly.
It exposes the alternative traffic ordering where a rear car can yield a gap
to the ramp; no trajectory, teacher label or Nash guarantee is supplied.

Seed4702, fixed 2,097,152 NEW joint transitions; 64 environments, rollout64,
five epochs, minibatch1024, gamma1/lambda1, all actor LRs1e-4, critic LR3e-4,
entropy.01, PPO clip.2, gradient norm.5. Sampling/minibatch RNG seed+451000000,
environment seed+450000000. This changes initialization AND training budget
relative to J, so it is not a single-factor causal attribution.

The SAME new single-PPO full-SUMO prerequisite must pass. New efficacy initials
95600001--95600128, half nominal/half dense, four modes; same nine methods,
**4,608 exact executions** with all three role reverts and explicit zero-update
joint initializer. Actual SUMO integration: **72 runs**, independently replayed.
If zero failures and positive joint gain/t lower versus strong team planning,
add the same six-method **3,072-run full SUMO** follow-up and replay. Otherwise
that stage is not enabled, and all negative results remain. Fixed final weights
only; no intermediate selection or dropped role/seed. No new statistical risk
certificate or goal completion follows from native pipeline completion.

Controller timing now includes the complete callback wall time, including
terminal exceptions. Calls exceeding500ms are counted descriptively, without
a hard-real-time claim. This instrumentation does not alter physical actions,
objectives, baseline algorithms or acceptance thresholds.
