# Development sizing for a new, fixed top-two MAPPO candidate

This rule is written AFTER reading the complete 320-initial fresh development
pilot. It is not blind to that pilot, does not reuse it as formal data, and does
not turn the previously failed reused-data screen into a pass. Freeze this rule
and its implementation before running the Monte Carlo calculation. No formal
attempt or statistical error is allocated by this calculation.

Keep all three actors 4703/4704/4705, TopTwoHistoryGate, the 0.0025 gain floor,
physical checks, forecast bank, phase/history tests, and strong baseline fixed.
For a potential attempt 6 use the unused error 0.05/42. The earlier allocations
0.025, 0.05/6, 0.05/12, 0.05/20, and 0.05/30 remain spent and unchanged.

The fixed design grid is calibration 4096/6144/8192 crossed with confirmation
1024/1536. Calibration is balanced across four plants with IID scene/mode draws
within each plant. Confirmation is IID from the original equal plant mixture.
Every design keeps all seven calibration and sixteen confirmation controllers;
cost is 7*Ncal + 16*Nconf. All original primary criteria and the stratified
joint-probability ratio bound remain unchanged. No early stopping on effects.

For each design use 4096 bootstrap datasets and deterministic seed stream
650401. Resample complete three-model rows, separately within each plant for
calibration and from the complete equal-mixture pilot for confirmation. Use
multinomial row counts; this is exactly equivalent to drawing rows with
replacement, including their paired gains and cross-model dependence.

Retain every empirical harm and failure. For calibration add independent
per-initial shocks at EXTRA conditional harm probabilities 0.003/0.005/0.01,
only on adopted rows not already harmful. Evaluate both a shock shared by all
models on each initial and shocks independent across models. These are explicit
planning assumptions, not estimates or guarantees of real future risk. The
middle stress 0.005 is fixed for selection; the other two are sensitivity
analyses. Prior formal attempt 5 observed roughly 0.003 conditional harm for
two models. Do not reuse its smaller joint-harm floor as if it were conditional.

Apply the existing alpha/8 per-component mixture ratio construction, exact
strong-lag and adoption tests, and paired-t gain lower bounds. All three models
must pass every primary statistical criterion in a synthetic dataset. Physical
failures are not injected synthetically; any actual pilot failure prevents
selecting a design. Monte Carlo CP lower tails are 0.01/(6*2) for the twelve
selection comparisons. Choose the lowest execution-cost design whose joint
pass lower bound exceeds 0.80 under BOTH 0.005 shock dependence models.
Publish every design/scenario, including a result of no qualifying design.
Do not enlarge the grid or change the rule after examining Monte Carlo results.

This calculation ignores remaining pilot-distribution uncertainty and cannot
guarantee 80% real power. A selected design still needs independent complete
preregistration, byte binding, actual-runtime preflight, an unused-slot audit,
public registration, and complete execution before it can support inference.
There is no communication-latency metric or hard-real-time claim.
