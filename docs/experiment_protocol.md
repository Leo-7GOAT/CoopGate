# Experiment protocol

The bundled pilot is an engineering study, not evidence of state-of-the-art
performance. The task is a short two-lane-to-one-lane merge, not a complete
highway on-ramp map. Three connected controlled vehicles interact with two SUMO
background vehicles. Logical main-lane indexing is maintained after the merge.

## Frozen choices

- Development pilot: seeds 7 and 19, 12,000 steps. Preliminary results revealed
  substantial policy seed variability. The final pilot extends both runs to a
  fixed 50,000 steps from scratch without checkpoint selection.
- Freeze actor weights before evaluation. No evaluation-based checkpoint search.
- Evaluate fresh held-out initial-state seeds beginning at 20,000 on nominal, dense,
  and scripted lead-braking variants; pair seed/scenario combinations across
  methods (the development pilot used seed 10,000 onwards). These variants also appear in training, so this is seed generalization,
  **not a claim of unseen-distribution robustness**.
- Count each episode once for collision/success metrics. Report each training
  seed separately before aggregate summaries. A small training-seed count limits
  any statistical conclusions.
- All methods have identical observations, vehicle constraints and speed MPC.
- There is no camera, perception network, V2X latency experiment, or implicit
  statement that traditional control is safe from every initial condition.

## What the initial comparisons can and cannot establish

The nominal/stress ablation isolates finite background hypotheses. Independent
versus joint checks expose the consequences of not rechecking a switched joint
action. MAMPS-inspired versus exhaustive switching uses matched acceptance
thresholds but changes search effort; it is an informative search comparison
but not an equal-compute ablation.

The rollout model is deliberately compact, approximates the MPC response and
SUMO lane changes, and includes a time-growing position margin. It checks an
initial candidate action followed by a rule-backup continuation. No invariant
terminal set is supplied. Therefore a positive margin is **model evidence over
a finite horizon**, not recursive safety or a verified feasibility certificate.

## Required before claiming a publication-level improvement

1. Reproduce a stronger MAMPS/MPS baseline with matched terminal conditions.
2. Calibrate/validate uncertainty using disjoint calibration and test episodes.
3. Include equal-risk and equal-compute comparisons; isolate threshold effects.
4. Test more training seeds, varied traffic densities and held-out behaviors.
5. Add continuous lateral vehicle-control validation (e.g. CARLA) if claiming
   full motion planning/control rather than traffic-level behavior switching.
6. Report failed and unclassified rollouts, emergency interventions, saturation,
   rare-event uncertainty, and all negative findings.
