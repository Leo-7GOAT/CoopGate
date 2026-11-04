# Completed fixed MAPPO gate study 2026-11-04

The complete fixed three-model family passes the original attempt6 joint
decision. This is evidence for the filtered closed-loop system on the declared
simulation distribution. It is not a universal safety certificate, a raw-policy
superiority result or a declaration that the paper is ready for a top venue.

## Actual execution and preserved registration

- Scientific source: `fe407b7c2199e9b72c4c1aa4cf2b47f96106fc1e`.
- Original registration: `2dc23bc91f63d877e98719b9543e5b3b29dfe5e6`.
- Recovery registration: `1e187a2a9e9e35886547f7261463e474e4bd0185`.
- 6,144 stratified calibration initials and 1,024 IID mixture confirmation
  initials; all 59,392 scheduled method-initial records completed.
- Recovery reused all 642 checksum-valid committed original records and
  executed 58,750 other planned records. All 59,392 records received the
  original independent controller/physics replay. No successful original
  root exit or count of lost in-flight attempts is invented.
- Recovery root PID 12208 was directly waited to actual exit0 on
  2026-11-04T08:58:19Z. All 16 collectors/reviewers, matching and finalizer
  have actual successful native receipts.
- Plan SHA256:
  `999a4fde61bc075a096a7f838d58365dd01ba0c5e9cc14d98fd9b7fb4e819a36`.
- Certificate SHA256:
  `1ca2500d3f88d8875affab0d3fdcdcd42447a0262452cf073e0b059b7bc96b25`.
- Comparisons SHA256:
  `54300c5b28fe583c64112396b835e8ac9bc13e10b64f22785b6edf253792fec9`.

All scientific thresholds and the already allocated level 0.05/42 remain
unchanged. Previous failed studies and their spent error remain preserved.
The interruption registration is separately
[public](https://github.com/Leo-7GOAT/CoopGate/releases/tag/mappo-attempt6-interruption-20261104).

## Genuine MAPPO and the candidate change

Each fixed seed 4703/4704/4705 contains three separate local actors and a
training-only centralized critic. Joint on-policy rollout retains sampled
proposals and their old log probabilities before physical filtering. Joint
GAE and clipped PPO updates train all three roles. This is warm-started MAPPO,
not training from scratch, and centralized filtering remains in deployment.
Neither three independent PPO runs nor a deployed global critic is substituted.

The fixed change integrates history/phase admissibility into candidate search,
so rejection of the first candidate does not discard other admissible options.
At most four additional repaired requests use local rank-two actor preferences;
the original candidates are retained. Actor weights, gain floor 0.25%, physics
and baseline are unchanged. The fresh development comparison evaluates this
whole change; it does not identify the separate causal effect of each part.

## Original joint decision

| Fixed model | Mean paired cost gain | Gain lower bound | Adoption / 1024 | Adoption lower bound | Mixture harm upper bound | Strong-lag harm upper bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4703 | 1.115133% | 0.832641% | 294 | 24.513347% | 3.008315% | 1.337217% |
| 4704 | 1.093357% | 0.813703% | 296 | 24.698771% | 2.927610% | 1.335241% |
| 4705 | 1.098964% | 0.814132% | 288 | 23.957663% | 3.360588% | 1.347180% |

The original thresholds are gain lower bound strictly above 0.5%, adoption
lower bound at least 20%, and both harm upper bounds at most 5%, for every
fixed model. Baseline/target have zero observed execution failures. The IUT
maximum component p value is 0.000008083334, below 0.05/42. The gain t bounds
are exact under normal IID sampling and otherwise asymptotic, not a general
finite-sample guarantee. Calibration is stratified, so its mixture bound uses
stratum joint-harm and adoption bounds, rather than a pooled binomial shortcut.

Harm means any vehicle's cost or arrival time worsens relative to its paired
baseline, or execution fails. It is not collision probability. Calibration
contains 5/5/8 harmful adopted episodes among 1819/1865/1835 adoptions; the
confirmation targets have 0/0/0 among 294/296/288. Zero observed confirmation
harm does not imply zero population risk or eliminate those calibration harms.

## Candidate-pool and adoption controls

All nominal and thinned controls use the same expanded candidate provider.
The thinned nominal controller removes confidence/history selection and uses
calibration-only independent episode thinning. Its actual confirmation
adoptions are 300/345/306, with 30/38/32 harmful adopted episodes and mean
gains 0.898465%/0.896183%/0.923027%. The gate has somewhat larger descriptive
mean gains and no observed confirmation harm. Matching is in expectation;
actual adoption differences are -0.586/-4.785/-1.758 percentage points and
the original intervals do not prove equality or equivalence. These descriptive
comparisons do not supply a new significance test or deployment certificate.

Unthinned nominal controllers have higher mean gains
2.512948%/2.413129%/2.529901%, but harm 93/97/90 adopted episodes. This shows
the efficiency/conservatism tradeoff; the gate does not maximize mean gain.
All 16 confirmation methods, including unfavorable results, remain in the
comparison file and manuscript. MPS/MAMPS entries are disclosed shared-oracle
raw-proposal adapters. They do not share the expanded candidate pool and are
not full reproductions or evidence of superiority over those publications.

## Engineering cost and interpretation

Target confirmation callback P95 is 164.681/186.243/163.445 ms, maximum
2624.294/1867.206/2068.470 ms. There are 254/273/254 callbacks over the 500 ms
control period. Timing includes the full actual work and preserves the
original/recovery batches. No communication-delay metric, timeout truncation,
hard-real-time claim or causal hardware speedup is introduced. The slow tail
is a material barrier to direct deployment.

The theoretical contribution is conditional: finite admissible-set inclusion
under the same extended state and complete prediction budget, and lag/bias
identifiability under explicit excitation/no-clipping/known-memory assumptions.
Neither implies closed-loop cost dominance or full future uncertainty coverage.
The retained exact counterexample explains why general varying innovations can
mask lag. The confidence score is a finite-path relative-gain score, not a
calibrated statewise probability that the action is safe.

The task remains three cars on fixed routes with known actuator classes and
synchronous state observations. Central-critic benefit versus IPPO, raw MAPPO
superiority over a strong planner, broad traffic generalization, independent
high-fidelity dynamics and real-vehicle validation remain unproved. These are
limits of the present evidence, not accomplishments to put in the abstract.

## Reproduce completed evidence

Run `python tools/mappo_completed_top_two_attempt6.py` after restoring all
standalone archive parts at the repository root. It binds the complete native
receipts, every scheduled sealed database, source snapshot and original
inference. Postprocessing integrity checks avoid redundant trace decoding only
after the mandatory original native replay has succeeded; no replay or sample
has been removed. The paper exporter then inspects all full trajectories for
exact baseline fallback, pre-adoption prefixes and actual per-role commands.
