"""Post-test mechanism analysis of EVERY raw timeout; no tuning or prevalence claim."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from coopgate.disturbance_v9 import DisturbedMergeEnv
from experiment_ego_v13 import run_episode
from study_independent_v18 import factories, verify_sources
from verify_development_v14 import trace_audit
from verify_single_v12 import equivalent


def replay(study, out):
    torch.set_num_threads(1); study, out = Path(study), Path(out)
    p = json.loads((study / 'protocol.json').read_text()); verify_sources(study, p)
    rows = json.loads((study / 'episodes.json').read_text())
    lookup = {(r['seed'], r['disturbance'], r['controller']): r for r in rows}
    failures = [r for r in rows if not r['completed']]
    assert failures and all(r['controller'].startswith('raw_') for r in failures)
    out.mkdir(parents=True, exist_ok=False); fs = factories(p['models']); cases = []; replays = 0
    with DisturbedMergeEnv() as env:
        for failed in failures:
            methods = ['baseline', failed['controller'], failed['controller'].replace('raw_', 'gate_')]
            case = dict(seed=failed['seed'], mode=failed['disturbance'], results={}, traces={})
            for method in methods:
                result, trace, initial = run_episode(env, failed['seed'], failed['scenario'], failed['disturbance'], fs[method], True)
                expected = lookup[failed['seed'], failed['disturbance'], method]
                for key in ('per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision', 'contract_violations', 'aborted'):
                    equivalent(result[key], expected[key])
                result.update(initial=initial.tolist()); trace_audit(trace, result)
                name = f"trace_{failed['seed']}_{failed['disturbance']}_{method}.json"
                (out / name).write_text(json.dumps(trace, indent=2) + '\n')
                case['traces'][method] = name
                case['results'][method] = dict(completed=result['completed'], arrival=result['per_vehicle_finish'],
                    cost=(np.array(result['per_vehicle_finish']) + .05 * np.array(result['per_vehicle_variation'])).tolist(),
                    adopted_steps=[i for i, t in enumerate(trace) if t.get('nominal_joint_action_admitted')],
                    terminal_state=trace[-1]['x'])
                replays += 1
            cases.append(case)
    record = dict(purpose='Post-test explanatory replay of every raw timeout; no new effectiveness test or model selection',
        episodes_sha256=hashlib.sha256((study / 'episodes.json').read_bytes()).hexdigest(),
        selected_by='All completed=false episodes, without outcome-dependent subsampling',
        additional_sumo_replays=replays, cases=cases)
    (out / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    plot(record, out)
    print(json.dumps(record), flush=True)


def plot(record, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(record['cases']), 2, figsize=(10.5, 3.2 * len(record['cases'])), squeeze=False, layout='constrained')
    colors = ['#777777', '#ce7549', '#287b8e']
    for j, case in enumerate(record['cases']):
        for color, (method, name) in zip(colors, case['traces'].items()):
            trace = json.loads((out / name).read_text()); t = np.array([a['t'] for a in trace]); x = np.array([a['x'] for a in trace])
            label = 'Baseline' if method == 'baseline' else 'Raw MARL' if method.startswith('raw_') else 'Joint gate'
            for column, role in enumerate((1, 2)):
                axes[j, column].plot(t, x[:, role, 0], color=color, label=label)
                axes[j, column].axhline(60, color='#999', linewidth=.5, linestyle=':')
                axes[j, column].set_ylabel('Route position (m)'); axes[j, column].set_xlabel('Time (s)')
                axes[j, column].set_title(f"Initial {case['seed']}, {case['mode']}: {'ramp' if role == 1 else 'main 2'}", fontsize=10)
                axes[j, column].grid(alpha=.15)
        axes[j, 0].legend(fontsize=8)
    fig.suptitle('All three observed raw-policy timeouts\nPost-test paired SUMO replays; gated models and thresholds unchanged', fontsize=12)
    fig.savefig(out / 'timeouts.png', dpi=160); fig.savefig(out / 'timeouts.pdf'); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('study'); p.add_argument('--out', required=True)
    a = p.parse_args(); replay(a.study, a.out)
