# Local control variation reserve development

For previous command p, current baseline command b and learning candidate u,
the triangle excess chi=0.05*(|u-p|+|b-u|-|b-p|) is nonnegative and equals
0.1*distance(u,[p,b]). It exactly measures extra variation cost of u followed
by b, relative to moving directly from p to b. It is zero between p and b.
The 0.05 weight is the ORIGINAL task cost weight, not a fitted safety margin.
This local identity is NOT a full future model-error or performance bound.

The new gate keeps the same three MAPPO actors, complete physical certificate,
candidate compositions, strict phase check, observed-history predicate and
0.0025 gain floor. For every complete finite forecast path it additionally
requires per-car predicted cost differences plus chi to be nonpositive and
the corresponding reserved mean relative gain to meet the same floor. A veto
continues candidate search. The baseline is selected and advanced exactly once.
No true parameters, hidden memory, future innovation, seed or case ID enters
the online gate. No existing controller source or raw observation is edited.

Before this comparison, six local math tests and three actual SUMO integrity
preflights with full controller/physics replay must pass. Reuse ALL100 earlier
development initials (the previous fresh80 are now reused development data,
plus all20 known adverse cases). Reuse the complete700 baseline/history/joint
observations and the three completed preflight reserve observations. Execute
only the remaining297 reserve episodes, all three actor seeds on every initial.
Four owned workers seal blocks and replay full actual controllers and physics.
Total effective paired observations are1000, not1000 new independent samples.
Do not stop early, drop adverse cases, allocate alpha or substitute a formal
deployment certificate. Report adoption, per-car harm, paired gains, exact
fallback and actual complete callback costs. Different collection batches do
not establish a causal wall-time speedup. No communication-delay metric.
