# Coupled MAPPO gate: current primary-source contribution boundaries

Checked2026-11-02 while attempt4 remains frozen and running. This note updates
the manuscript's related-work framing; it does not change any method, sample,
allocation or comparator in the registered experiment. This is a targeted
review, not an exhaustive novelty proof or evidence of superiority.

## Closest prior objectives

| Primary source | What the original work already covers | Required limitation on our claims |
|---|---|---|
|[Cho et al., CSPI-MT,2024](https://arxiv.org/html/2408.12004)|Statistical testing of threshold policies against a baseline, multiple cutoff candidates and selection power under controlled erroneous adoption.|A confidence threshold, multiple candidates and a reliable baseline fallback are not new by themselves. Our paired simulator outcomes are not a reproduction of its observational-value estimation pipeline.|
|[Banerjee et al., DMPS,NeurIPS2024](https://papers.nips.cc/paper_files/paper/2024/file/b589d92785e39486e978fa273d0dc343-Paper-Conference.pdf)|Reward-aware recovery planning uses learned long-term values; the paper analyzes safety and recovery regret.|Do not claim that considering performance during shielding is new, or that our finite-path condition inherits its recovery-regret theorem. We have not run its full planner/training method.|
|[Brorholt,Larsen,Schilling,AAMAS2025](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p399.pdf)|Assume-guarantee reasoning composes individual shields into a global safety result, without online communication.|Our local MAPPO actors plus centralized joint gate do not constitute a fully decentralized shield. Three cars do not establish large-agent scalability. We do not reproduce this synthesis procedure.|
|[Frauenknecht et al.,UPSi,2026](https://arxiv.org/html/2604.26836)|Probabilistic neural-network ensemble uncertainty is propagated through reachable sets with an explicit certainty constraint; practical approximations are evaluated in model-based RL.|Our known-model finite witness bank is not a rigorous reachable-set overapproximation of its complete parameter/noise domain. Uncertainty-aware filtering and maintained RL performance are established topics.|
|[El Yamani,Krasowski,Vanneaux,September2026 preprint](https://arxiv.org/html/2609.07618)|Decentralized predictive shields combine separately pretrained agents, finite-horizon model-based Q-learning during deployment and communication-free conflict resolution, targeting group cumulative reward with safety.|Do not claim the first performance-aware multi-agent predictive shield. Our joint CTDE training, fixed deployed policies, centralized gate and any-car regression event are different settings. This distinction is an analysis of objectives, not a claim of measured superiority.|

The latest UPSi page identifies version3 dated2026-08-10 and its RLC2026/
Reinforcement Learning Journal publication. The decentralized shielding source
is arXiv2609.07618v1 dated2026-09-07; no accepted-venue status is inferred.
Only the primary papers linked above support these source summaries.

## What the present implementation can substantiate

The learning component is genuine MAPPO: a central training critic and three
local actor updates from joint on-policy rollouts. This implements an existing
algorithm; it is not a newly invented MAPPO variant. Its initialization ancestry
must remain disclosed. The old actor-level MAPPO/IPPO evidence does not prove
a consistent centralized-critic advantage or generic raw-MAPPO superiority.

The proposed study asks whether a **frozen full joint decision policy** can
retain useful actual learning adoption and paired team gains while controlling
the probability that **any vehicle** loses cost or arrival performance versus
the same executable feedback baseline, conditional on at least one genuine
learning adoption. The physical stopping contract and statistical performance
statement are different guarantees. A finite-path confidence score is not a
per-state probability of safety.

The newly measured mechanism is a prediction gap: midpoint/noisy paths unioned
with corner/clean paths omit parameter/noise interactions. An exhaustive
declared-class coupled prefix probe exposes complete first-action regressions
in28/29 historical harmful model-initial incidents. A compact fixed set of1–3
extra paths per uncertain class is fitted to those failures. Its complete665-run
development study reduces known regression incidents29 to3 while preserving
useful adoption/gain on80 new balanced initials. These are measured development
facts with a fitted regression set, not an independent population risk claim.
See [complete data and limitations](mappo_compact_coupled_results_20261102.md).

Scenario Cartesian products, greedy set cover, minimum-over-paths scores and
baseline-advantage telescoping are established mathematical tools. Their use
alone does not establish top-venue novelty. The defensible method argument
must connect the concrete coupled prediction error to retained per-car checks,
an exact rejection-only implementation, independent trajectory-level evidence
and measured full callback costs. The [mechanism note](mappo_coupled_mechanism_20261102.md)
states the algebra and the missing uniform-error assumption explicitly.

## Required final-paper wording and comparison scope

Keep the complete negative attempt3 result and all remaining development
failures. Read attempt4 only after actual native completion and report its
original joint decision unchanged. Report same-adoption controls as matching
in expectation, with actual discordance intervals; no equality/equivalence
claim follows automatically.

MPS/MAMPS controls in the registered study are shared-physical-oracle adapters,
not full original-paper reproductions. DMPS,UPSi,the AAMAS compositional method
and the September2026 decentralized method were not executed in this study.
Do not state state-of-the-art superiority over them. Do not silently add,
replace or drop registered comparators after outcomes become available.

Repeated multisecond/over500ms callbacks prevent a hard-real-time assertion.
Known-route longitudinal SUMO experiments do not prove perception robustness,
arbitrary intersections, arbitrary vehicle counts, communication performance,
sim-to-real transfer or road-vehicle safety. A stronger practical-deployment or
novelty claim would require additional targeted evidence; registration success
alone cannot establish either. The final manuscript must reflect the actual
completed result and these limits, whether the joint decision passes or fails.
