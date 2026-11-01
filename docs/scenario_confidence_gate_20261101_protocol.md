# Explicit confidence-score threshold, with independent risk certification

The complete three-model MAPPO integration retains1,200 actual SUMO runs,
all28,118 commands/140,590 physical substeps and identical strong baselines.
Bank gains for4703/4704/4705 are .4846%/.6423%/.5536%, with29/28/30 adopted
conditions and zero observed harmful adoptions. These repeat the SAME FIVE
initials and do not certify risk or satisfy all-model utility acceptance.
The original minimum finite-bank margin is ZERO at all adopted decisions of
the examined4703 controller:175 nominal and47 bank decisions. Unchanged
coordinates/time buckets make that minimum degenerate; it is not useful
calibrated confidence and was never justified as a probability.

Prepare a SEPARATE gate with an explicit deterministic scenario pass score.
Retain the same local MAPPO actors,baseline,candidates,projection,strict
nominal Pareto/mean-gain ranking,all33 prediction paths,reference model and
actuator stopping certificate. A path passes iff every vehicle's cost/time
and registered interpolated goal-arrival comparison passes the original
thresholds. Score=passing paths/33. Thresholds are1,32/33,30/33, encoded as
allowed failed-path counts0,1,3. A failed or incomplete forecast still rejects
the candidate globally; only complete performance-regression paths consume
the new scenario budget. More than the allowed count rejects early. A
selected candidate still has complete evidence for all33 paths.

This changes performance admission, not the physical safety certificate.
At budget0 it must reproduce the original bank controller exactly. At larger
budgets some modeled performance regressions are explicitly possible. Score
0.95 is NOT95% safety probability: these33 fixed paths are not IID samples
from the real plant distribution. Every selected score,threshold,violation
count and path difference must be logged; the old risk certificate remains
invalid for the new controller. Never relabel this as the unchanged bank gate.

First validate score/budget behavior using explicitly synthetic forecast
fixtures and compare budget0 with actual original predictions across all
four classes. These tests are not traffic efficacy evidence. Then register
actual-SUMO smoke/development collection and a new checker that verifies
bounded actuator memory,full physical recurrence,all33 complete score
comparisons and the whole gate replay. The original strict-bank reviewer
cannot validate intentionally allowed positive differences and must not be
bypassed by mislabeling the new rule.

Before formal calibration, lock all three fixed MAPPO models,score thresholds,
new independent initial distribution,calibration/confirmation split and fresh
familywise error allocation. Keep risk<=5%,adoption>=20%,utility>.5% unchanged
for EVERY model. No threshold is certified on the five used development
initials and no best seed is selected. Confirmation must evaluate the entire
fixed controller on new independent trajectories,not treat paths/decisions
or repeated plant/mode conditions as independent trials.

Include an adoption-matched control chosen using calibration only. One
specified candidate is state-independent episode-level thinning of the
nominal-performance selector,with an independent controller RNG and all
actual callback costs charged separately; its expected adoption must match
the score gate under calibration,then both rates are reported on confirmation.
Any source/threshold change invalidates a frozen confirmation. IPPO,online
cost and paper novelty review remain separate required work. This prototype
is not a new risk certificate,first-ever switching algorithm or completed paper.
