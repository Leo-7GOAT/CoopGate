# Replicate the fixed L joint MAPPO configuration

L/4703 completed real joint MAPPO, all role/GAE audits,5,120 exact comparisons,
80 actual SUMO integration/replay runs and a separate400-run actuator gate
integration. Front/rear policy substitution shows behavioral contribution,
but raw team advantage versus the strong team planner is NOT established.
Bank-gated integration gains .4846%, below the unchanged .5% utility threshold,
and its repeated five-initial outcomes do not certify conditional risk.

Register joint seeds4704 and4705 as exact replications of L's training
configuration, not additional hyperparameter searches: each2,097,152 new
joint transitions, all three independent ReLU256 local actors updated,
central critic, gamma/lambda1, entropy.01, LR[1e-4,1e-5,1e-4], critic3e-4,
64 environments/rollout64/five epochs/minibatch1024/logitScale5/clip.2.
Both start from the SAME completed `single_4601` actor set and the same
disclosed broad rear-coasting initialization. They use independent new
proposal/environment RNG streams, seed+451000000/seed+450000000.

These are independent JOINT optimization seeds conditional on one fixed
pretrained parent; they are not independent end-to-end single/joint training
pipelines or a from-scratch study. The separately registered4602/4603 single
replications remain required and unchanged;4603's failed confidence-interval
condition is retained. No original three-seed prerequisite is declared passed.

Each final fixed model is compared on128 fresh initials and four modes,
4704:96000001--96000128;4705:96100001--96100128. Ten methods include rule,
both strong planners, parent single PPO, final joint PPO, untouched joint
initializer, all three original-role substitutions and the highest-admissible
parent rear substitution:5,120 exact executions EACH. Then80 actual SUMO
integration runs EACH and full independent physical/controller replay.

Only zero final failure plus positive raw team gain and positive development
one-sided t lower versus strong TEAM enables that model's prospectively
specified3,072 full SUMO follow-up. Otherwise record it as not enabled, retain
all raw failures, and never retry/select an intermediate checkpoint. The
same model-independent conditions apply to both seeds. No best-seed choice.

Both native pipelines may run concurrently on separate CPU cores while4602's
SUMO prerequisite completes. Timing is explicitly not an isolated benchmark.
Do not alter their frozen source/protocol dependencies until both terminate.
Three completed joint runs would establish optimization replicates, not
automatically superior learning, a new gate certificate or paper readiness.
Gate calibration, same-adoption controls, IPPO, runtime and paper revision
remain separate required work. Prior PI experiments and risk budgets stay intact.
