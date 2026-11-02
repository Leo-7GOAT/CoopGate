# Preserve the original acceptance contract for the new MAPPO family

This records requirements,not an allocated experiment or a certificate.
The original authoritative definitions are `recovery_risk_v38.py`,
`recovery_risk_v39.py` and `design_joint_risk_v42.py`. Their fixed old model
IDs and spent allocations must never be relabeled or reused for MAPPO.

The new fixed family is4703/4704/4705. Every model must satisfy BOTH mixture
and strong-lag conditional regression risk<=.05,mixture adoption LOWER
confidence bound>=.20,and paired percentage team-cost gain LOWER one-sided
Student-t bound STRICTLY>.5%. A point gain>.5% is insufficient. Zero observed
baseline/candidate failures is also required. Strong-lag risk is not dropped
when the mixture result looks favorable. Keep the original nominal Pareto
ranking,physical class/memory assumptions and per-vehicle harm definition.

Harm includes controller failure OR baseline failure OR ANY vehicle's paired
cost/arrival regression>1e-9. Retain failed prefixes,capped costs,zero-adoption
failures and baseline-unassessable samples. Impute baseline failure
conservatively and block acceptance; never remove the sample. For successful
zero-adoption episodes,require physical/outcome equality to the paired baseline.
Do not use the development-only checker that stops on a recorded collision as
the sole formal outcome collector: it preserves evidence but cannot certify
a complete fixed-N study after stopping. Distinguish inconsistent evidence
from an accurately recorded adverse outcome in the formal implementation.

Sampling unit: one independent initial assigned ONE scene,ONE plant and ONE
disturbance before execution. Calibration retains original balanced plant
strata and within-stratum IID draws; confirmation retains original IID full
mixture without balancing/rejection. Reference sizes are3072 calibration and
1024 confirmation,not a smaller convenient grid. Repeated paths,decisions,
models,shards or disturbance reruns never enlarge independent N.

Conditional mixture risk uses the original stratified joint-harm/adoption
bounds; do not pool balanced-stratum adopted observations as a binomial.
Strong-lag risk and IID-mixture adoption use exact binomial inference. Paired
gain uses the original t bound: normal-IID exact,otherwise explicitly
asymptotic with finite variance. All models/targets must pass,so a fixed
intersection-union claim has max-component p-value. Selecting among thresholds
is a DIFFERENT union and needs multiplicity protection or an untouched
independent selection/validation split. Do not confuse these two operations.

Before launch,audit the project's existing attempt/error ledger,prove the
preceding attempts terminal and bind an unused summable allocation. Do not
reset total error to.05,borrow a withdrawn trial's allocation,or choose an
attempt index without checking its prior use. These allocations are NOT
assigned in this requirements note. Lock source/model hashes,threshold family,
random streams,schedule,full expected counts and all acceptance tests first.
Confirmation runs to its full registered scope even if calibration fails;
retain negative results. No continuation based on a promising partial effect.

Same-adoption comparison must use an actual independently randomized nominal
selector control fixed using calibration only. Charge each full online callback
separately,including terminal failures; do not share an offline evidence budget.
Report realized adoption differences and their uncertainty in confirmation,
not assume expected-rate matching implies exact matching. Independent controller
randomness cannot use future innovations,true actuator parameters or environmental
seed information as an online feature. Communication latency is outside scope.

The current800-execution development grid is NOT any of these formal samples.
It can diagnose behavior and score tradeoffs,but cannot supply the above
certificate or claim the manuscript ready for submission.
