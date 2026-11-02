# Actual compact coupled-gate development result

All **665 actual SUMO executions** completed with independent exact controller
and physics replay. Four owned child launchers and the root launcher actually
exited 0. All raw paired blocks are sealed. No observed execution/physical
failure occurred. In 356 successful zero-adoption episodes, commands, states
and times exactly equal the independently executed baseline trajectory.

## 80 fresh balanced development initials

Each row is one unchanged MAPPO training seed evaluated on the same 80 initials.
Percent gain is mean paired team cost gain versus the strong baseline, including
episodes with no adoption. Harm is any-car cost/arrival regression, not collision.

| Seed | Original harmful/adopted | Coupled harmful/adopted | Original gain | Coupled gain |
|---|---:|---:|---:|---:|
|4703|0/25|0/25|1.014006%|1.014006%|
|4704|0/22|0/22|1.176843%|1.176843%|
|4705|0/25|0/24|0.980691%|0.978965%|

Coupled adoption is 31.25%, 27.50%, 30.00%. These 80 initials are development
data, not a formal or independent per-state risk probability certificate.

## All 15 known adverse initials

| Seed | Original harmful/adopted | Coupled harmful/adopted | Original gain | Coupled gain |
|---|---:|---:|---:|---:|
|4703|12/14|2/9|1.940076%|0.954779%|
|4704|8/15|0/10|2.359746%|1.394252%|
|4705|9/13|1/10|1.999399%|1.068632%|

Across model-initial incidents, observed harmful adoptions fell **29 to 3**
(89.66% reduction) on this deliberately selected regression set. This is not a
population risk reduction estimate: the extra witnesses were fitted using
these adverse cases. All original failures and the failed formal study remain.

Two adverse initials remain: nominal/dense/uniform seed 852000369 for 4703
(ramp cost +0.15625), and gain_drag/tight_headway/bias seed 853000735 for
4703 and 4705 (rear cost +0.0625). Arrival differences are zero. The second
case shows why rejecting a stored first action does not prove whole-episode
repair: the modified gate can choose a different candidate or a later action.

## Actual full callback time

| Stage | Seed | Original P95 | Coupled P95 | Coupled max | Coupled calls >500 ms |
|---|---:|---:|---:|---:|---:|
|Fresh|4703|124.485 ms|120.205 ms|890.578 ms|9|
|Fresh|4704|138.963 ms|135.943 ms|896.800 ms|6|
|Fresh|4705|124.406 ms|132.509 ms|933.988 ms|9|
|Regression|4703|283.570 ms|317.669 ms|1102.796 ms|6|
|Regression|4704|328.201 ms|341.911 ms|1087.987 ms|8|
|Regression|4705|284.653 ms|306.517 ms|1092.544 ms|5|

These wall times include the complete callback under four concurrent study
workers. They do not establish a speedup, worst-case execution bound or hard
real-time control. No communication latency was studied.

## Decision and next step

The unchanged MAPPO weights plus 1–3 additional coupled paths per uncertain
class are a promising **development candidate**: regression harm decreased
substantially and new-case adoption/gain remained useful. Preserve this exact
candidate rather than fitting away the remaining cases. Before promotion,
audit that attempt4 is unused, register its independent sample/complete control
family and spend only its new scheduled error (0.05/20), then execute the full
unchanged mixture risk <=5%, adoption lower >=20%, paired gain lower >0.5%
and zero observed execution-failure requirements. Do not transfer attempt3's
certificate, recycle its alpha or call these development outcomes passed
formal validation. The overall project goal remains active.

Machine-readable source: `development/compact_coupled_validation_20261102/`.
Restore its complete raw study from the byte-verified completed delivery ZIPs.
