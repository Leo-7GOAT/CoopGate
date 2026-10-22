"""Freeze measurable improvement criteria before any new held-out simulation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.merge_v4 import MergeConfig

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v8'


def main():
    D.mkdir(exist_ok=True)
    if (D/'protocol.json').exists():
        raise RuntimeError('Protocol already frozen')
    seeds = np.random.default_rng(2026101528).choice(np.arange(11000000, 12000000), 200, replace=False).tolist()
    files = ['src/coopgate/fast_v8.py', 'src/coopgate/merge_v4.py', 'src/coopgate/planning_v5.py',
             'src/coopgate/learning_v6.py', 'src/coopgate/verify_v7.py', 'src/coopgate/experiment_v8.py',
             'tools/study_v8.py', 'tools/cold_start_v8.py']
    files += [f'artifacts/v6/seed{s}/policy.pt' for s in (201, 211, 223)]
    result = {'version': 'v8', 'config': MergeConfig().dict(), 'model_seeds': [201, 211, 223],
              'test_seeds': {'nominal': seeds[:100], 'dense': seeds[100:]},
              'methods': ['constant', 'reference_pair', 'fast_pair', 'fast_full', 'bb_natural', 'bb_learned', 'reference_full'],
              'executions': 1400, 'retraining': False,
              'development_data': 'All 360 previously published V6/V7/V7-pair test states reused only for implementation audit, never counted as new independent evidence.',
              'order': 'Per-scene method permutation from default_rng(seed+808); disclosed raw execution_order.',
              'timing': 'Single serial worker, warmed kernel and model; selection plus first simulator step included in first response. Fresh-cache process startup separately measured.',
              'criteria': {'zero_recorded_safety_violations': True, 'all_fast_pair_costs_equal_reference_pair': True,
                           'all_full_and_pruned_selections_equal_reference_full': True,
                           'fast_pair_first_response_p95_ratio_max': .2,
                           'bb_learned_first_response_p95_vs_reference_pair_max': .2,
                           'bb_learned_mean_certificate_reduction_vs_fast_full_min': .3},
              'bound': 'Independent earliest arrivals under maximum jerk-admissible acceleration and speed saturation; conflicts removed. Prune only lower_bound > incumbent+1e-10. Preserve ties by original candidate index.',
              'limits': ['Exact frozen-model finite-portfolio optimum, not global driving optimality.',
                         'Compiler speedups are engineering gains; do not attribute them to learning.',
                         'Warm latency is not startup latency, hard real-time or delayed-plant validation.',
                         'The statistical confidence-score limitation from V7 is not solved by faster model verification.'],
              'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    (D/'protocol.json').write_text(json.dumps(result, indent=2)+'\n', newline='\n')


if __name__ == '__main__':
    main()
