# Audited MAPPO deployment into actuator recovery gates

This independent development integration retains the exact V34 physical
baseline, its class bounds and command-history memory interval. It accepts
only completed, audited joint MAPPO with actual updates to all three actors.
It never transfers the earlier PI controller's risk certificate.

`tools/evaluate_mappo_recovery.py` collects and reviews separately. The caller
must observe actual zero native collection exit before review. Every raw
episode, terminal callback exception and capped failed outcome is persisted
before physical checks. Review reconstructs every physical substep, every
baseline command and the entire memoized gate callback, including its work
counters; it also replays terminal exceptions. Do not retry into an existing
output directory, truncate traces or omit failed outcomes.

The numerical smoke is five actual SUMO executions: one original nominal
initial, nominal plant, clean innovation, and all five rules. Use seed96700001.
It tests wiring, serialization and independent replay, not efficacy.
An initial smoke of the completed K actor does not select K as the final
method or establish learned cooperation superiority. L continues unchanged.

Full integration uses all five original scenes, four actuator classes, four
innovation modes and all five rules (baseline, MPS-inspired, MAMPS-inspired,
nominal performance, finite-bank performance). For n=1 and seed_base96800001,
this is400 actual SUMO executions on five independent initial draws. The
scene-index offset is1000; repeated classes/modes do not inflate independence.
Method order rotates and reverses per paired condition. The same fixed model
must be named and audited before collection; no best-checkpoint selection.

Every selector has its own complete callback cost, including baseline,
candidate construction, forecasts and terminal failures. No shared evidence
work budget is reused. Warm-up is separately excluded. Concurrent workloads
must be disclosed; even isolated p95/max measurements are not hard-real-time
guarantees. Communication latency is outside scope as requested.

Report all failures, baseline failures, adopted episodes, conditional harmful
episodes, every vehicle's cost/arrival, all-scope gains only when all pairs
complete, complete-pair descriptions, callback p95/max/sum and >500ms. Zero
adoption gives an undefined conditional-harm estimate, not zero risk.
Finite-bank margin is not a probability. These descriptive repeated outcomes
cannot establish fresh joint risk/adoption/utility acceptance, matched-adoption
superiority or general OOD safety. Those remain separate prospective studies.
