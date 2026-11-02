# Online observation consistency for actuator lag uncertainty

The reviewed20-initial/140-SUMO scalar implementation study retains one strong-
lag clean counterexample shared by all three MAPPO models. Privileged OFFLINE
baseline-advantage decomposition proves its first adopted decision causes
the missed per-car regression despite all41 existing forecast paths passing.
Its second adoption partly repairs cost;the vector telescoping identity has
zero numerical residual. No true coefficients,memory or future tape may enter
an online repair. More threshold tuning on these cases is not confirmation.

Prepare a separate32-cell retention-parameter set-membership observer for the
known unit-gain/zero-drag mild/strong lag classes. Inputs: decision-boundary
observed position/speed and actually executed prior command ONLY. Maintain
outward interval actuator memories per cell,propagate5 physical substeps using
bounded innovations and velocity clipping,and discard cells inconsistent with
the next observed position OR speed (fixed1e-9 additional tolerance). Never
silently reset an empty set. Returned bounds enclose the retained union and
are NOT posterior probabilities. Gain/drag uncertainty remains unsupported.

The deterministic containment argument is induction: the true initial memory
is zero;each interval update encloses every rho/memory/innovation in its cell;
the actual observation lies in its containing cell's propagated interval,so
that cell cannot be removed under the declared assumptions. Exact sensing,
fixed dynamics,noise bounds and valid reset are required. A future prediction
bank made from retained endpoints would STILL be finite scenario evidence,
not full-set performance robustness. No baseline/candidate performance or risk
certificate follows merely from this observer property.

First verify containment with bounded innovations,velocity saturation,zero
speed,changing actual commands and contradictory measurements. Then replay
the already verified adverse/other traces to measure informativeness without
claiming new samples. A prospective gate may use retained parameter bounds
and reconstruct candidate memory from command history;that full controller
needs its OWN actual SUMO tests,whole-callback timing,same-adoption controls
and all unchanged formal risk/adoption/utility requirements. No new gate is
implemented or promoted by the observer-only study. All real MAPPO actors,
critic/PPO audits and every failed earlier result remain unchanged.
