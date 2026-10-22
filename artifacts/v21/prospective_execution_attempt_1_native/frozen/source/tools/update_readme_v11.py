"""Refresh the top-level current result without rewriting frozen history."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    r=json.loads((ROOT/'artifacts/v11/summary.json').read_text())
    a=r['overall']['smooth_learned'];p=r['contrasts_perturbed']['all']['v10_rollout']
    old=r['perturbed']['v10_rollout'];new=r['perturbed']['smooth_learned']
    t=p['time'];v=p['variation'];learning=r['contrasts_perturbed']['all']['smooth_constant']['time']
    counts=t['better_tied_worse_clusters'];ci=t['paired_cluster_ci97_5'];vc=v['paired_cluster_ci97_5']
    passed=sum(r['criteria'].values())
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    history=readme[readme.index('## V10:'):]
    top=f'''# CoopGate

**Reproducible cooperative merge control in Eclipse SUMO.** The current V11
controller combines a frozen learned profile proposal, joint backup safety
checks, and online planning. Earlier MAPPO/MPC experiments remain available
below; the V11 learner is a centralized contextual bandit, not newly trained
distributed MARL. This is a research prototype with conditional model-based
guarantees, not a certified road-driving system.

## V11: faster merging with lower command variation

V10 optimized arrival time alone and increased average command changes relative
to V9. V11 uses a **lexicographic arrival/total-variation objective**: time first,
then smaller changes on time ties. Commands lie on a fixed 0.625 m/s² lattice,
so both costs are integers. A hold-enabled feedback family supplements the
original binary proposal/brake family; each receives a coordinate rollout.
Every trial retains the unchanged joint V9 robust backup certificate.

The code, ten criteria and **200 fresh initial states × four error modes × nine
methods** were frozen in commit `7fc085e` before 7,200 actual SUMO executions.
**{passed}/10 criteria passed.** Arrival-time reduction was **{t['reduction_percent']:.4f}%**,
below the prespecified 1% target; this criterion remains **unmet**. No threshold
was revised after testing. Both co-primary intervals exclude zero. The two co-primary
endpoints compare V11 with the frozen V10 controller on identical states and
exogenous disturbance tapes.

| Mean over 600 perturbed runs per method | V10 | V11 | Reduction |
|---|---:|---:|---:|
| Capped mean arrival time | {old['mean_cost_s']:.3f} s | {new['mean_cost_s']:.3f} s | {t['reduction_percent']:.2f}% |
| Three-car command total variation | {old['mean_total_variation']:.3f} m/s² | {new['mean_total_variation']:.3f} m/s² | {v['reduction_percent']:.2f}% |
| Mean command change rate | {old['mean_command_jerk']:.3f} m/s³ | {new['mean_command_jerk']:.3f} m/s³ | {p['jerk']['reduction_percent']:.2f}% |

The paired 97.5% seed-cluster intervals for the mean reductions are
[{ci[0]:.3f}, {ci[1]:.3f}] s and [{vc[0]:.3f}, {vc[1]:.3f}] m/s².
Both co-primary endpoints receive this prespecified confidence-level allocation;
bootstrap inference is approximate. The 600 conditions contain **200 independent
initial-state clusters**, not 600 independent scenes. Total variation is not
divided by episode duration, preventing slower driving from diluting it.

The complete controller finished **{a['completed']}/800** runs with
{a['collision_episodes']} recorded collisions, {a['contract_violation_episodes']}
route-violation episodes, {a['aborted']} aborts and {a['timeouts']} timeouts.
Warm controller-cycle P95 is **{a['warm_controller_cycle_p95_ms']:.2f} ms**;
warm first-response P95, including initialization and the first TraCI cycle, is
**{a['warm_first_response_p95_ms']:.2f} ms**. All failures remain in the data.

There are still adverse outcomes: versus V10, perturbed arrival time improves
in {counts[0]}/200 clusters, ties in {counts[1]}, and regresses in {counts[2]}.
The nominal theorem is a **lexicographic bound versus the base policies**,
not componentwise dominance, not a guarantee versus V10, and not a realized
performance bound under disturbances. Independent paired experiments test the
two improvements. Command variation is not measured physical jerk or passenger
comfort.

Matched ablations remove the variation objective, the hold family, or the
second family sweep. No-learning, fixed and random proposals isolate the
learning contribution. Learning's extra arrival gain over no-learning is
{learning['reduction_percent']:.2f}% (descriptive 95% interval for seconds saved:
[{learning['paired_cluster_ci95'][0]:.3f}, {learning['paired_cluster_ci95'][1]:.3f}]).
The online planner's entire gain must not be attributed to reinforcement learning.

The scope remains **three cooperative fixed-route vehicles, exact measurements
and a known ±0.5 m/s² actuator-error bound**. SUMO pauses during computation;
warm timing quantiles do not establish worst-case or delay-aware real-time
performance. Rollout, safety filtering and lexicographic optimization are prior
art; this work provides a concrete construction, conditional proofs and frozen
comparative evidence. Published-method comparisons and broader traffic settings
remain necessary for a strong paper submission.

[V11 results and ablations](docs/v11_results.md) · [Theory and proofs](docs/theory_v11.md) ·
[Frozen protocol](artifacts/v11/protocol.json) · [All runs](artifacts/v11/study/episodes.csv) ·
[Fixed trajectory example](artifacts/v11/trajectory_example.png) ·
[Unmet criterion and historical regressions](docs/v11_acceptance.md) ·
[Related work and novelty limits](docs/related_work_v11.md)

```powershell
pip install -e ".[dev,fast]"
python tools/verify_release_v11.py --replay
python tools/replay_v11.py --gui --disturbance bias --out runs/my_v11_demo
python tools/study_v11.py --out runs/my_v11_reproduction
```

The verifier checks provenance, exact accounting, statistics and replays;
passing it does not turn a failed performance target into a success.
`python tools/verify_release_v11.py --require-acceptance` additionally enforces
all ten performance criteria and intentionally exits nonzero for this release.

'''
    (ROOT/'README.md').write_text(top+history,encoding='utf-8',newline='\n')


if __name__=='__main__':main()
