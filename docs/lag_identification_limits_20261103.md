# What the observed-history fit can and cannot identify

The conditional inverse result in
[lag_inverse_identifiability_20261103.md](lag_inverse_identifiability_20261103.md)
requires a constant innovation within the five-substep control window. It is
not a proof for arbitrary bounded innovations. This note makes the distinction
constructive,without modifying the fixed gate or the running study.

Let h=0.1,with unit gain,zero drag,known m0,and constant command u. For arbitrary
substep innovations b_s,the same algebra yields

\[
Z=0.01(m_0-u)\phi(\rho)+0.01\sum_{s=1}^{5}(3-s)b_s.
\]

If b_s is constant,the last term vanishes. If b_s=b_bar+delta_s with
|delta_s|<=epsilon_b,the residual can instead be as large as0.06epsilon_b.
Consequently,the earlier inverse sensitivity denominator cannot justify a
point estimate under changing innovations. With observation error eps_Z,
known m0 and a valid monotone interval with |phi'|>=c,the conditional inverse
error bound becomes

\[
|\hat\rho-\rho|\le
\frac{\epsilon_Z+0.06\epsilon_b}{0.01|m_0-u|c},
\]

provided the reported root really solves the corresponding constant-bias
equation within that interval. Current grid/local least-squares fitting is not
such a certified root solver,so this is an explanatory bound,not its guarantee.
If initial memory is uncertain by eps_m,using observed memory m_hat adds
0.01eps_m max_I|phi| to the numerator and replaces |m0-u| by |m_hat-u|.

A stronger obstruction is an exact observation-equivalence example. Set
m0=0,u=1,rho_A=3/4 and rho_B=19/25. Both rho values belong to the declared
strong-lag class. In plant A set all b_A,s=0. In plant B set

\[
b_{B,s}=m_{A,s}-m_{B,s}
        =\rho_B^s-\rho_A^s.
\]

Every innovation is below the admitted0.1m/s² magnitude. Both plants then
have identical m_s+b_s,so all five speeds and positions coincide,not merely
the boundary observation. Yet their hidden retention values differ. No
estimator using those observations and commands alone can distinguish these
two admissible parameter/innovation realizations within this window.

The executable proof uses exact rational arithmetic and checks all five
identical physical states and innovation bounds:

```powershell
python tools/verify_lag_identification_limits.py --out artifacts/mappo_ctde_20261101/lag_identification_limits_20261103
```

It is a model counterexample,not an empirical frequency claim about the
random innovation generator. It leaves constant-bias identifiability intact.
It explains why the fitted history path is additional finite prediction
evidence rather than an actuator posterior or complete future uncertainty
coverage. Physical recovery assumptions and independent closed-loop
performance validation remain separate obligations. No policy,threshold,
source snapshot,independent initial or error allocation changes here.
