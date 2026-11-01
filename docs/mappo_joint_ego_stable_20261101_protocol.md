# Joint MAPPO development L: preserve pretrained ego competence during cooperation

K completed 2,097,152 joint transitions, full audits, 4,608 exact comparisons,
72 SUMO integration runs and replay. Team mean remains .4050% worse than strong
team planning (paired difference -.0260s; development t lower -.0909s), while
ramp cost is1.1611s worse. The large follow-up was not enabled by its prospective
positive-development condition. All outcomes remain. Actor1 squared parameter
drift is9.4426 versus .0303 during the successful single stage; this is a
diagnostic association, not proof that drift causes the performance gap.

L starts from the SAME fixed new single-PPO parent `single_4601/policy.pt` and
the same explicitly disclosed broad rear-coasting initial prior as K. The ego
actor learning rate is1e-5, ten times smaller than front/rear actors1e-4. Critic
LR3e-4. ALL THREE actors continue receiving newly collected on-policy PPO
updates; none is frozen and no teacher loss/PI update occurs. The intent is to
retain the already demonstrated ramp skill while other roles adapt. This does
not guarantee monotonic improvement or learning superiority.

Seed4703, fixed 2,097,152 NEW joint transitions, 64 vector environments,
rollout64, five epochs, minibatch1024, gamma1/lambda1, entropy.01, clip.2,
gradient norm.5, sampling RNG seed+451000000/environment seed+450000000.
Architecture, physical filter, training objective, strong baselines and old
formal thresholds remain unchanged. Seed also differs, so no isolated causal
claim is made for the learning-rate change from K.

Fresh efficacy initials95700001--95700128, half nominal/dense, four modes;
same nine policies including untouched initializer and all three role reverts,
PLUS a rear-role replacement with the unchanged highest-admissible parent
actor. This avoids interpreting recovery from an always-coasting rear actor
as evidence of learned conditional yielding. **5,120 exact executions**, 128
independent initials. **80 actual SUMO integration executions** and full replay.
Only zero failure plus positive raw
joint gain AND positive development t lower against strong TEAM planning
enables the preregistered **3,072-run full SUMO** follow-up and replay.
Otherwise retain the negative result and record that stage as not enabled.
Fixed final weights only, no checkpoint cherry-picking or seed deletion.
New multi-seed/gate/actuator/risk/adoption/runtime validation remains separate;
completion of this native queue never marks the research goal complete.
