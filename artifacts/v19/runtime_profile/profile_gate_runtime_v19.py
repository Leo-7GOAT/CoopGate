"""Post-audit runtime attribution on two slow recorded decisions per model."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import shutil

import numpy as np
import torch

from coopgate.bounded_portfolio_v19 import BoundedJointPortfolioGate
from coopgate.cooperative_iteration_v16 import load_coupled
from probe_disturbed_gate_v19 import digest, save

ROOT = Path(__file__).resolve().parents[1]


def profile(directory):
    torch.set_num_threads(1)
    audit = ROOT / 'artifacts/v19/bounded_gate_audit'
    plan = json.loads((audit / 'plan.json').read_text())
    assert (audit / 'sha256.json').exists()
    for path, expected in json.loads((audit / 'sha256.json').read_text()).items():
        assert digest(audit / path) == expected
    episodes = json.loads((audit / 'episodes.json').read_text())
    lookup = {(r['initial_index'], r['disturbance'], r['controller'], r['implementation']): r for r in episodes}
    candidates = {seed: [] for seed in plan['models']}
    for ci, (index, mode, name) in enumerate(plan['cases']):
        row = lookup[index, mode, name, 'bounded']
        state, previous = np.array(row['initial']), np.zeros(3)
        trace = json.loads((audit / f'trace_{ci}_bounded.json').read_text())
        for step, item in enumerate(trace):
            candidates[name.rsplit('_', 1)[1]].append((item['gate_ms'], ci, step, state.copy(), previous.copy()))
            state, previous = np.array(item['x']), np.array(item['command'])
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for seed, metadata in plan['models'].items():
        assert digest(ROOT / metadata['path']) == metadata['sha256']
        model, _ = load_coupled(ROOT / metadata['path'])
        BoundedJointPortfolioGate(model).select(warm, np.zeros(3))
        for old_ms, ci, step, state, previous in sorted(candidates[seed], key=lambda v: (-v[0], v[1], v[2]))[:2]:
            gate = BoundedJointPortfolioGate(model)
            gate.step = step
            profiler = cProfile.Profile()
            command, info = profiler.runcall(gate.select, state, previous)
            stats = pstats.Stats(profiler)
            records.append(dict(seed=seed, case_index=ci, step=step, recorded_ms=old_ms,
                                profiled_ms=info['gate_ms'], command=command.tolist(),
                                selected_composition=info['composition'],
                                calls=sorted([dict(file=Path(file).name, line=line, function=function,
                                                   calls=values[1], self_ms=1000 * values[2], cumulative_ms=1000 * values[3])
                                              for (file, line, function), values in stats.stats.items()],
                                             key=lambda r: -r['cumulative_ms'])[:25]))
    report = dict(purpose='Post-hoc attribution on six slow decisions; not a runtime distribution or an efficacy study',
                  source_audit_sha256=digest(audit / 'audit.json'), records=records)
    save(output / 'profile.json', report)
    shutil.copy2(__file__, output / Path(__file__).name)
    save(output / 'sha256.json', {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps([dict(seed=r['seed'], case=r['case_index'], step=r['step'], ms=r['profiled_ms'], calls=r['calls'][:8])
                      for r in records]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    profile(args.directory)
