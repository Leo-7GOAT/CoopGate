# Grouped validation rejects promotion of the imitation gate

The fixed73-128-128-29 supervised gate training produced six leave-one-initial-
group-out models and five final ensemble members. An independent audit checks
all11 checkpoint SHA256 hashes,training-only fold normalization and masks,
and recomputes all registered fold selections. All5,964 runtime feature vectors
match the offline dataset byte for byte. Original training terminal exit was
not recovered; the audit explicitly records this missing native receipt.

At threshold.90,the only fold selecting positive classes selects25 decisions,
with ZERO correct teacher classes. At.95,it selects12,again ZERO correct.
At.99,all six folds select zero positive decisions. The remaining folds also
select zero positives at.90/.95. Baseline-heavy overall accuracy does not
demonstrate useful arbitration. There are just SIX independent initial groups,
despite5,964 correlated rows and135 positive labels.

This result FAILS the positive-generalization prerequisite. No fast student is
promoted into a SUMO controller,no high-softmax safety claim is made,and no
teacher or older policy certificate transfers. Final ensemble weights remain
as reproducible failed-study artifacts,not as a selected successful model.
No traffic efficacy or risk inference is drawn from classification alone.

MAPPO proposal actors remain the separately audited centralized-critic,
three-local-actor on-policy policies4703/4704/4705. This experiment is plainly
supervised gate imitation and is never counted as additional PPO training.
Any future imitation experiment needs materially more INDEPENDENT initial
groups and fresh grouped/closed-loop validation; more correlated frames do
not repair this evidence deficit.

Evidence: `artifacts/mappo_ctde_20261101/fast_gate_training_20261102/` contains
the plan,all11 weights,fold report and `independent_audit.json`.
