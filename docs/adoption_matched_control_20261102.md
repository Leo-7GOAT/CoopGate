# State-independent thinning as the adoption-matched control

Implementation is prepared,not launched or calibrated on development outcomes.
Use complete independent calibration cases for the same fixed model to estimate
target episode adoption a_G and nominal-selector adoption a_N. If0<a_N and
a_G<=a_N,freeze p=a_G/a_N. If a_G>a_N,thinning cannot match the target and must
be reported infeasible,not silently capped at1. Zero nominal adoption does not
identify a probability. Current five-initial development counts are NOT the
formal calibration sample and are not used to freeze p here.

Before confirmation,draw one controller coin C~Bernoulli(p) per independent
episode from an explicit study/controller stream,independent of scene,plant,
observed state,environmental seed and all innovations. Store the full coin
schedule/hash before collection. C=true runs the complete nominal-performance
selector; C=false runs the complete common baseline. Each episode computes
only its selected branch. Charge its full callback cost,including failures;
no cached shared evidence or hypothetical uncomputed work is charged instead.

Let A_N,H_N be the potential adoption/harm events under the nominal selector.
Actual learning adoption of this control is A_T=C AND A_N,not C alone. For
fixed p>0 and independent C,

P(A_T)=p P(A_N), and P(H_T | A_T)=P(H_N | A_N).

Thus matching adoption by random abstention does not create a better risk
discriminator;it is the required control against explaining every gate benefit
by lower adoption. These identities are conditional on the fixed calibration-
chosen p. Estimated rates do not guarantee equal true or realized adoption.
Report confirmation adoption differences/uncertainty and all conditional harm
denominators. At p=0,conditional harm is undefined and the20% adoption target
cannot pass. Preserve failures before first adoption and baseline failures.

All three fixed models remain. Each model may have its own preconfirmation
p estimated from the SAME calibration distribution,never selected on holdout
effects or to discard a seed. Balanced calibration plant weights stay those
of the original mixture;confirmation remains IID mixture. Original mixture/
strong-lag risk,adoption lower and gain lower requirements are unchanged.
This control has no new certificate and does not replace fresh calibration,
confirmation,multiplicity/error-ledger allocation,full physical replay or
realized online cost evaluation.
