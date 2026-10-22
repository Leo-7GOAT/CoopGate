"""Predeclare V7 before accessing new calibration or test outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.merge_v4 import MergeConfig

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v7'


def main():
    D.mkdir(exist_ok=True)
    if (D/'protocol.json').exists():
        raise RuntimeError('Protocol already frozen')
    seeds = np.random.default_rng(2026101517).choice(np.arange(8000000, 9000000), 1184, replace=False).tolist()
    files = ['src/coopgate/merge_v4.py', 'src/coopgate/planning_v5.py', 'src/coopgate/learning_v6.py',
             'src/coopgate/risk_v7.py', 'tools/data_v7.py', 'tools/calibrate_v7.py', 'tools/study_v7.py',
             'artifacts/v6/protocol.json', 'artifacts/v6/calibration.json']
    files += [f'artifacts/v6/seed{s}/policy.pt' for s in (201, 211, 223)]
    result = {'version': 'v7', 'config': MergeConfig().dict(), 'model_seeds': [201, 211, 223],
              'calibration_seeds': {'nominal': seeds[:512], 'dense': seeds[512:1024]},
              'test_seeds': {'nominal': seeds[1024:1104], 'dense': seeds[1104:]},
              'thresholds_s': np.arange(-3., .01, .25).tolist(), 'risk': .1, 'delta': .05, 'tolerance_s': 1e-6,
              'methods': ['constant', 'learned', 'v6_conformal', 'robust_conformal', 'robust_pac', 'teacher'],
              'executions': 960,
              'selection': 'Maximize accepted calibration count among thresholds with CP upper <= risk in BOTH regimes; stricter threshold wins ties. Reject all if none.',
              'correction': 'One-sided Clopper-Pearson delta/(2*13) per fixed regime and threshold.',
              'conformal_baseline': 'Max of independently computed nominal/dense one-sided residual quantiles at alpha=.1.',
              'scope': 'Conditional accepted nonbaseline performance risk, with 95% confidence over calibration, for each declared regime and any mixture of them. Not arbitrary OOD or collision risk.',
              'data_cost': '1024 SUMO initializations plus two-profile model rollouts (baseline and selected proposal); not full simulator episodes.',
              'test_protocol': '80 matched scenes per regime; all six methods execute in real SUMO. Full teacher costs used for evaluation only. First scene of each regime traced for every method.',
              'frozen_policy': True, 'retraining': False,
              'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    old = json.loads((ROOT/'artifacts/v6/protocol.json').read_text())
    assert not set(seeds).intersection(s for group in [*old['data_seeds'].values(), *old['test_seeds'].values()] for s in group)
    (D/'protocol.json').write_text(json.dumps(result, indent=2)+'\n', newline='\n')


if __name__ == '__main__':
    main()
