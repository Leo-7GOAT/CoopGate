# Prospective fast arbitration surrogate: supervised gate,unchanged MAPPO

The parameter gate's fresh development shows multi-second worst callbacks.
Prepare a lightweight offline teacher surrogate without replacing,retraining
or renaming any of the genuine MAPPO actors. Inputs are the existing73
online-only features. Labels0..28 encode baseline or teacher composition/
repair choice. They are imperfect teacher decisions,NOT safe/no-harm labels.
No teacher risk/performance certificate transfers to a student controller.

Fixed architecture73-128-128-29 ReLU;AdamW3e-4,decay1e-4,100 epochs,batch256,
gradient cap5,class weights sqrt(n_baseline/n_class) capped10. Mask classes
unseen in that training partition. Fit normalization on TRAINING rows only,
std floor.05. Leave each of six independent initial groups out,keeping ALL
models/plant/mode frames of that initial outside training. Report positive
precision/recall and selected counts,not only inflated baseline-class accuracy.
Then fit five full-data seeds22011..22015 with fixed budgets. No best epoch,
seed or group choice. All fold failures remain. No new simulator trial occurs.

Prospective runtime score is mean class softmax with unanimous ensemble label,
thresholds.90/.95/.99. This is imitation confidence,NOT safety probability.
Runtime baseline must be the unchanged physical baseline;candidate decoding
must use the same physical composition/projection/certificate. Missing or
uncertain labels fall back. Never run the expensive future-forecast teacher
inside the claimed fast callback. Whole callback costs must be measured.

Feature/decoder parity and grouped validation are prerequisites,not traffic
evidence. Register actual SUMO smoke and held-out closed-loop tests separately.
Every fixed MAPPO model and threshold must remain visible. Fresh complete-
controller calibration/confirmation,unused error allocation,same-adoption
controls,risk<=5% for mixture AND strong lag,adoption lower>=20%,gain lower>.5%
and original failure retention remain required. No formal sample is collected
or certification assigned by this training experiment.
