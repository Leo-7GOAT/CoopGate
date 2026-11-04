# Fresh development follow-up: complete results

The complete fresh pilot contains 320 independent development initials and
2240 actual SUMO episodes. All four workers and their native owner exited 0;
all raw blocks were sealed and every controller/physics trajectory replayed.
An independent analysis reconstructed all 960 paired model-initial outcomes
from the 2240 sealed records and exactly reconciled the original statistics.

| Fixed actor seed | Old/new adopted episodes | Old/new mean paired cost gain | Change, percentage points | Old/new harmful episodes |
|---|---:|---:|---:|---:|
| 4703 | 86 / 97 | 1.099836% / 1.351273% | +0.251438 | 0 / 0 |
| 4704 | 91 / 105 | 1.110316% / 1.372274% | +0.261957 | 0 / 0 |
| 4705 | 86 / 97 | 1.176738% / 1.410362% | +0.233624 | 0 / 0 |

Every baseline and target execution completed without a recorded physical or
controller failure. Harm means any car's paired cost/arrival regression, not
just a collision. 1358 successful zero-adoption model episodes across both
gates exactly reproduced the paired baseline's commands and physical traces.
Zero observed harm is not a population-risk certificate.

Old is ObservedHistoryGate. New is TopTwoHistoryGate: it combines integrated
candidate-level admissibility search with bounded local rank-two expansion.
The fresh comparison estimates the benefit of that WHOLE change, and does
not separately identify the contribution of search integration and rank-two
expansion. Earlier reused-case experiments and all unfavorable comparisons
remain available; the earlier prespecified reused-data screen still FAILED.
No new formal statistical error was allocated to any of these development runs.

The three independent actors are the existing genuinely trained MAPPO actors;
their centralized critic is used only during training. No new model training
occurred in this follow-up. Preference logits are not feasibility probabilities.
This does not establish a causal advantage of a centralized critic over IPPO.

Actual complete callback P95s for the new gate were 138.573/159.994/137.984 ms;
maxima 1703.693/1766.598/1656.935 ms, with 46/48/42 callbacks exceeding 500 ms.
These include the full online selection and fallback work. There is no
cross-batch timing-speedup, communication-delay metric or hard-real-time claim.

The raw archive has 984 files, 46,997,311 original bytes and two independently
extractable ZIP parts totaling 37,816,874 bytes. Every archived member was
read back and SHA256 checked. Extract both parts at the repository root.
`completed_top_two_fresh_delivery_20261104/inventory.json` binds each raw byte.

The separately registered post-pilot sizing calculation selected 6144 new
calibration and 1024 new confirmation initials, with all original criteria and
controls preserved. Its 36 Monte Carlo scenarios are planning assumptions,
not actual SUMO experiments or guarantees of real statistical power. Attempt 6
still needs its own allocation, exact source/preflight binding, public
registration and complete untouched-data execution. Formal attempt 5 remains
complete and FAILED; this development result does not replace that decision.
