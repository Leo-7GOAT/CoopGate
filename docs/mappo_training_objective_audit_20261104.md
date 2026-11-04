# Training objective and observation audit

This note identifies a possible next training change, not an implemented new
policy, a deployment certificate, or an explanation proven by an ablation.
The fresh bounded-candidate pilot is now complete and supports testing the
fixed proposal change independently; the training change below remains unimplemented.
All existing models, frozen sources, failed screens and error allocations stay.

The actual joint reward in src/coopgate/cooperative_v15.py, step(), is negative
mean individual stage cost plus potential shaping. The stage cost is active
time plus0.05times actual command variation. src/coopgate/mappo_ctde.py,
PPOVector.step(), retains that reward for joint training. All three local actors
share this payoff through the centralized critic. This is genuine MAPPO, but
the reward does not explicitly penalize a positive relative regret for any one
vehicle against the paired strong baseline.

For example, per-car cost changes[-0.2,-0.2,+0.1]improve team mean by0.1while
violating the actual any-car performance criterion. Thus a better training
team reward alone cannot prove the deployment criterion. This algebraic
counterexample establishes an objective distinction, not the cause of every
observed harmful episode. Full controller/physics diagnosis remains necessary.

One candidate training objective would minimize E[J_team]+lambda*E[H], where H
is the exact paired terminal event used for performance evaluation, including
any cost or arrival regression and physical failures. The paired baseline can
be simulated under the same initial condition, plant and innovation tape in
the TRAINING environment. No such true plant, future tape or counterfactual
ground-truth cost may become an actor or online-gate input. Lambda should be
fixed within a rollout/PPO update and recorded with reward traces. This is a
proposal for constrained training, not a completed algorithm or a convergence
claim. In particular, an unconditional training E[H]constraint does not certify
the later deployment Pr(H|A), because filtering changes the trajectory and the
conditioning event. Existing independent deployment validation remains required.

The present batched training transition also differs from the matched actuator
validation model. CooperativeVector.step() uses batch_step and additive errors;
validation introduces declared gain/drag and actuator memory classes. A future
training environment should match those transitions before claiming robustness
across them. The current actor is feed-forward over present local features;
public class indicators and measured local history are possible inputs. Actual
hidden memory and true coefficients remain privileged training/evaluation data.

For a fixed actuator retention, holding a command for the0.5sdecision interval
gives m[t+1]=q*m[t]+(1-q)*u[t], q=exp(-0.5/tau). Reconstructing from only H past
commands with zero initial memory produces error q^H*m[t-H]. Because actual
memory remains within[-5,2.5]and tau<=0.5s in the declared classes,
|error|<=5*exp(-H). For H=8this is approximately0.001677314m/s^2.
This bound concerns the omitted initial-memory tail for the SAME fixed true
coefficient. It neither identifies the unknown coefficient nor bounds a neural
network's estimation error or arbitrary future innovations. It motivates a
finite observed-history representation without claiming that history alone
makes the partially observed game fully observable.

Constrained multi-agent learning is prior work: Gu et al.'s
[Multi-Agent Constrained Policy Optimisation](https://arxiv.org/abs/2110.02793)
introduces MACPO and MAPPO-Lagrangian. A Lagrangian penalty is therefore not a
new contribution by itself. Bianchi et al.'s
[Scalable Safe Policy Improvement for Factored Multi-Agent MDPs](https://proceedings.mlr.press/v235/bianchi24b.html)
already addresses multi-agent baseline improvement with structured joint-action
selection. Belouadah et al.'s
[CSPO](https://proceedings.mlr.press/v306/belouadah26a.html)
studies constraint-sensitive primal-dual updates. These primary abstracts were
checked on2026-11-04; complete method adaptation and numeric comparisons have
not been performed. None of their stated guarantees transfers automatically
to CoopGate or to the unimplemented training proposal above.

If the fresh pilot fails to provide consistent useful improvement, the next
concrete implementation should be a separate actuator-matched, baseline-relative
joint MAPPO training prototype with explicit on-policy proposal/GAE checks for
all three actors. Begin with a short integrity budget and independent physical
replay before a full three-seed training run. Do not modify the existing frozen
trainer, relabel an existing checkpoint, or spend formal6error on an unvalidated
prototype. Raw-policy competence, gated competence, same-adoption comparisons,
online costs and untouched formal evidence remain separate obligations.
