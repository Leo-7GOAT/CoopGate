# Rejection-only optimistic bound: independent development prototype

The reviewed L recovery integration has41 bank-gate callbacks>500ms. This
prototype accelerates the SAME admission rule by proving some forecast
candidates inadmissible before computing their remaining baseline continuation.
No running MAPPO study imports the new module and no old source is edited.

For one fixed reference-model forecast prefix, dominate every future command
by the maximum slew-admissible acceleration, remove nonnegative drag, ignore
traffic constraints, and propagate memory/speed/position upward using explicit
nextafter(+infinity). Retention is in[0,1), gain positive, speed clipped to
[0,20]. Thus this relaxed trajectory is optimistic. Its decision-rounded goal
arrival is a LOWER bound for the original continuation's goal arrival. Add
only already incurred variation to obtain a lower bound on full per-car cost.
This bound uses only a controller-owned fixed prediction tape, not realized
future innovations or true actuator parameters.

Memory domination follows from nonnegative retention/complement and command
domination. Removing nonnegative drag enlarges acceleration. Floating-point
multiplication/addition with nonnegative factors is monotone, and upward-rounded
intermediate products/sums preserve an upper envelope. Clipping is monotone.
Induct on substeps to bound position. Earlier envelope crossing gives an
optimistic decision-boundary arrival; cumulative absolute variation cannot
decrease. Therefore lower cost/time above the paired reference limits proves
that this candidate cannot pass the original per-vehicle admission comparisons.

Check this at future decision boundaries, after the proposed first command.
Only prune if a lower bound exceeds the reference by1e-8 (more conservative
than the existing1e-10 admissibility tolerance). Baseline forecasts stay full.
Every potentially admissible candidate still receives the full original
forecast, terminal checks, physical certificates, ranking and all required
prediction paths. No bound can authorize learning. Pruning error code4 means
proved performance rejection, not an incomplete physical safety proof or a
simulator/controller failure. Log complete rejection-prefix witnesses and
actual bound work separately from planning transitions.

First test independent dynamics with maximum-acceleration and random slew-
admissible sequences across all four classes; test exact original results with
pruning disabled and prove selected pruned examples fail full original checks.
Then numerical smoke replays both original and new whole gate for the four
nominal/dense, nominal-plant, clean-mode performance-selector source episodes.
This selection is development coverage including a known slow case, not a
representative efficacy sample. Preserve a failed smoke and fix only in a
separately named follow-up, never overwrite its evidence.

If smoke passes, FULL paired replay covers all160 completed original
performance-selector episodes, both implementations=320 whole-controller
replays. Compare every actual command, baseline evidence, adoption event,
selected composition/weight, accepted path differences and final margin.
Rejected-candidate diagnostics/work can differ by design. Both implementations
receive their own full measured callbacks, with rotating/reversing order and
separate caches. Warm-up is excluded, no cost is shared. Retain every exception
and mismatch before aborting. Full replay has no added SUMO executions or
independent initials and issues no new risk or hard-real-time certificate.

Counterbalanced timing may overlap registered MAPPO training; disclose this
and treat speedups as paired development measurements. Fresh actual SUMO and
runtime validation remain required before replacing a published controller.
The unchanged risk<=5%, adoption>=20%, utility>.5% research requirements and
all original results/failures remain intact. Do not claim this standard
admissible-pruning principle alone is a new research theorem.
