# Completed powered MAPPO attempt 5

The fixed joint deployment decision FAILED. All scientific thresholds and all
three seeds remain. This is a complete negative result, not an incomplete run.

| Model | Mixture harm upper | Strong lag upper | Adoption lower | Mean paired gain | Gain lower |
|---|---:|---:|---:|---:|---:|
| 4703 | 4.804782% | 2.685338% | 20.220066% | 0.916485% | 0.669764% |
| 4704 | 4.831272% | 2.178094% | 19.853707% | 0.915582% | 0.666125% |
| 4705 | 5.064290% | 2.712785% | 20.403414% | 0.908674% | 0.663255% |

Required: risk upper <=5%, adoption lower >=20%, gain lower strictly >0.5%,
zero observed execution failures for baseline and targets. Joint p is
0.0024355897497639455; allocated alpha=0.05/30 is SPENT. Seeds4704 and4705
fail different conditions. Zero observed failures does not mean zero risk.

The primary study completed45056 actual SUMO episodes and full independent
controller/physics replay:4096 stratified calibration initials and1024 IID
confirmation initials. All16 collect/review child exits, finalizer and actual
native root exits are0. Scientific source76d3a772336e0ffd7a568ff2d63a01ebd4ca9a26
and registration99afdd57422c0079a7f71d370f3d8fe921398fa1 were public before
collection. None of the163 registered sources changed. A complete165-file
snapshot is included in the raw archive.

Both registered1024-initial three-model ablations completed:6144 EXTRA actual
SUMO episodes, no new independent initials or alpha. The compact comparison
adopts291/298/292 episodes versus246/242/248 for history. Harm remains2/1/1
for BOTH gates; history loses0.046570/0.088687/0.041950 gain percentage points.
The phase-only comparison also removes NO additional harmful episodes through
history. These results do not validate the proposed history mechanism's benefit.

Full diagnosis retains14 harmful model-initial incidents on8 unique initial
conditions. All11513 successful zero-adoption trajectories exactly equal the
actual baseline; all15360 pre-first-adoption prefixes and3072 randomized
comparison branches match their independent executions. These trajectory checks
are not a theorem of full continuous-parameter performance coverage.

The verified12-page Chinese manuscript has9tables,12editable native equations
and3actual-data figures. Every final page was rendered using owned hidden Word
and inspected. All preserve-only reference package parts and page geometry
remain. The older drafts are retained. The manuscript explicitly reports the
negative decision, unfavorable controls and >500ms callbacks; it is not a
submission-ready claim of state-of-the-art performance.

Primary archive:2143files,647254032 original bytes,584011019 ZIP bytes,23
standalone byte-verified ZIPs. Registered supplements:836files,115632814 original
bytes,94076921 ZIP bytes,4standalone ZIPs. All27 remote asset SHA256 digests
have been verified in the draft release. Final public availability must be
confirmed by the separate publication receipt; a draft is not public delivery.
Extract every ZIP into the repository root; the supplemental asset filename
prefix does not affect its unchanged internal paths. Existing random-proposal
control raw data remain public in completed_random_proposal_delivery_20261103.

Three independent local actors and centralized training critic are genuine
MAPPO. All roles have nonzero learning updates and actual execution changes.
Raw MAPPO's general superiority over the strong planner and a causal central
critic advantage over IPPO remain unproved. Shared online gating is centralized.
Training uses a batched numerical environment; validation actually uses SUMO.
The safety certificate is conditional on the declared model class and geometric
contract. Confidence scores are finite prediction margins, not probabilities.
