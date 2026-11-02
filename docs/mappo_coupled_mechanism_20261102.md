# Why the coupled witness repair has a physical and mathematical rationale

This note explains the new development controller. It is not a successful risk
certificate, a full uncertainty-box proof or a claim that new scenes passed.
The actors remain the three genuine CTDE MAPPO policies from the completed study.

## 1. The omitted interaction is in the plant equation

The simulator and frozen prediction kernel use, for each vehicle,

\[
m_{t+1}=\rho m_t+(1-\rho)u_t,\qquad
v_{t+1}=\operatorname{clip}\{v_t+\Delta t[g m_{t+1}-d v_t^2+w_t]\}.
\]

The old finite bank took the **union** of midpoint-parameter/noisy paths and
parameter-corner/clean paths. A union does not represent their Cartesian
product. Under braking, the derivative of applied acceleration with respect
to gain has the sign of negative memory, while its derivative with respect
to drag is always nonpositive. Pairing low gain with high drag and high gain
with low drag omits independent gain/drag corners. Persistent innovation can
also change later speeds, drag, discrete planner choices and arrival times.
Thus the effects cannot generally be added after two separate tests.

The exhaustive prefix probe uses all independent parameter vertices crossed
with constant bias sign vertices. Reconstructing memory from executed commands
and a declared retention point uses past information only. These paths expose
complete cost/phase regressions that the old union missed in 28 of 29 harmful
model-initial incidents. All 28 are uncertain-actuator incidents. One nominal
uniform-noise incident is still uncovered. This is a development diagnosis;
covering its stored first actions does not establish closed-loop improvement.

## 2. A finite bank remains finite evidence

For a fixed augmented state z, candidate a and parameter/noise path s, write

\[
\widehat\Delta_i^s(z,a)
=\widehat J_i^s(a\!\to\! B\mid z)-\widehat J_i^s(B\mid z),\qquad
q(z,a)=\min_s\frac{-\operatorname{mean}_i\widehat\Delta_{i,\mathrm{cost}}^s}
 {\operatorname{mean}_i\widehat J_{i,\mathrm{cost}}^s(B\mid z)}.
\]

The denominator must be finite and positive. Admission retains every original
per-car cost/arrival test, the 0.05 s crossing-phase test, the physical stopping
certificate and q >= 0.0025. Adding paths can only reduce q **for that same
candidate at that same state**. Early rejection using a partial minimum is
exact: a partial minimum upper-bounds the eventual minimum. It cannot remove
a candidate that would satisfy the full minimum threshold.

This is neither a calibrated probability nor a vertex sufficiency theorem.
Nonlinear drag, clipping, discrete planner switches and discretized arrival
times invalidate an unsupported claim that extrema occur at parameter vertices.
There is also no implication that adding paths monotonically lowers the
whole-episode conditional harm rate: candidates, later states and the set of
episodes with any adoption can all change.

## 3. What a true performance guarantee would require

Augment z with time, previous command, actuator state/observer and baseline
controller state. For each car, let V_B,k(z) be the **true** remaining capped
cost of the unchanged baseline, including any terminal abort accounting. With
the same future exogenous disturbance realization in paired counterfactuals,
define the true baseline advantage

\[
A_{B,i,k}(z,a)=c_{i,k}(z,a)+V_{B,i,k+1}(z')-V_{B,i,k}(z).
\]

Telescoping along the executed hybrid trajectory gives

\[
J_i(\pi)-J_i(B)=\sum_k A_{B,i,k}(z_k,a_k).
\]

When the hybrid executes B at its current augmented state, that term is zero.
Nonpositive true advantages at every learned intervention would therefore
give a per-car cost guarantee. The finite bank estimates these counterfactual
advantages; it does not know them. An error bound on each estimate, if proved
under valid regularity assumptions, would translate into an accumulated
performance bound. No such uniform bound is asserted here. The new witnesses
target a measured source of estimation error, while fresh trajectory-level
calibration is still necessary.

## 4. Engineering tradeoff to measure

The full offline library has 512 paths for gain/drag and 64 for either lag
class. The fixed compact controller adds 3, 1 and 2 paths respectively to the
original bank, chosen by deterministic first-prefix witness coverage across
all three actors. It adds none for nominal parameters. Original paths, order,
weights, threshold and physical baseline remain. All admitted candidates must
pass the extra paths; no hidden realized parameter or future tape is consumed.

Whether this compact representation preserves useful adoption and gains while
reducing harmful interventions is the purpose of the independently frozen
665-run **development** experiment. Full callback P95, maximum and >500 ms
counts are measured. Communication latency and hard real-time/road-vehicle
guarantees are outside this experiment. An eventual formal experiment must use
an untouched sample and unused, independently audited error allocation.
