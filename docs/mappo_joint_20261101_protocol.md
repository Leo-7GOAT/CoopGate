# Three independently parameterized actor MAPPO development, J

This is a separate new MAPPO experiment. V14/V16 PI results and their
certificates remain attached to their old controllers. No unchanged PI policy
is renamed MAPPO. The new stage is PI-warm-started on-policy MAPPO, with the
pretraining budget and unchanged initializer disclosed separately.

## Prerequisite and fixed training

Start only after the NEW single-PPO I run's complete 2,560 actual-SUMO
executions on 128 fresh initials/four modes have a positive ramp cost gain
and development one-sided t lower >0 versus the unchanged strong ego planner,
zero recorded execution failures, and complete independent physical/controller
replay. This is a single-seed development prerequisite, not the original
three-seed V14 acceptance or a distribution-free risk certificate.

Parent: fixed final `single_4601/policy.pt`, 262,144 actual PPO transitions.
J seed4701, 1,048,576 NEW joint simulator transitions; 64 vector environments,
64-step rollouts, five epochs, minibatch1024, gamma1/lambda1, actor LR1e-4,
critic LR3e-4, entropy.01, gradient clip.5, PPO ratio clip.2. Three ReLU256 local
actors have independent parameters. The Tanh256 centralized team critic uses
all current public kinematics/command history/clock. Sampling logits are scaled
by five; initial main-road preference after scaling is only .4*LEVELS.
Proposal/minibatch RNG resets to seed+451000000 after construction, environment
RNG is seed+450000000. No teacher labels or PI updates occur during PPO.

Unlike the single stage, ALL active roles' sampled proposals now enter the
actual joint transition and clipped-PPO losses. The common reward is negative
mean per-car arrival/variation cost plus the original terminal-zero potential.
Log probabilities are pre-filter sampled proposals. The decentralized actor
does not read critic input/output. The shared physical filter remains central.
This does not imply an equilibrium, monotonic improvement or road safety.

## Complete prospective development comparison

Fresh initials95500001--95500128; first64 nominal, remaining64 dense; each has
the four original disturbance modes. Compare unchanged rule, strong ego,
strong team, parent single PPO, untouched joint initializer, final joint MAPPO,
and three final joint models each reverting exactly one role to its initial
parameters. All others continue reacting closed-loop. **4,608 exact-simulator
executions**, 128 independent initials; role variants add no independent N.
The initializer is explicitly zero-update and never counted as trained MAPPO.

Review both parameter changes and effective behavior. At least an additional
role must demonstrably affect cooperation before attributing gains to learned
multi-agent coordination. A role whose optimal action remains unchanged need
not be forced to change. No policy-gradient theorem is claimed from a positive
parameter delta or from this finite role-replacement experiment.

Actual SUMO integration uses the same nine methods on two declared kinematic
test seeds95500001/95500002, one nominal/one dense, four modes: **72 runs** and
full independent replay, not extra independent efficacy initials. If the final
joint policy has zero failures, positive team gain and development one-sided t
lower >0 versus STRONG TEAM planning, additionally execute all six non-reverted
methods on the original full128-case distribution in SUMO: **3,072 runs** and
complete replay. Otherwise this large follow-up is not enabled, and all negative
exact results/integration outcomes remain. This is a prospective development
resource condition, not a formal final-test selection procedure.

Every callback/time/failure in the original tools remains recorded. Results
are descriptive development evidence; formal multi-seed confirmation, actuator
deployment, gate risk/adoption/utility validation and matched-adoption/runtime
comparisons remain separate requirements. Existing .5% utility/20% adoption/5%
regression-risk criteria are NOT relaxed or transferred to J.

## Native execution

`tools/run_joint_mappo_stage.py` binds source/parent/proof hashes before starting,
retains direct child handles, records actual native zero/nonzero exits and
stops on operational failure without retry. It does not kill existing jobs,
read other desktop projects, delete data, change checkpoints or shut down.
The development outcome can be scientifically negative even when all native
commands exit zero; pipeline completion is not goal completion.
