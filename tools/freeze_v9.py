"""Freeze perturbations, controls and material-improvement targets before test."""
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.merge_v4 import MergeConfig
from coopgate.experiment_v9 import METHODS

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v9'


def main():
    D.mkdir(exist_ok=True)
    if (D/'protocol.json').exists(): raise RuntimeError('Protocol already frozen')
    seeds = np.random.default_rng(2026101609).choice(np.arange(13000000, 14000000), 200, replace=False).tolist()
    files = ['src/coopgate/robust_v9.py', 'src/coopgate/disturbance_v9.py', 'src/coopgate/experiment_v9.py',
             'src/coopgate/fast_v8.py', 'src/coopgate/merge_v4.py', 'src/coopgate/planning_v5.py', 'src/coopgate/learning_v6.py',
             'tools/study_v9.py', 'artifacts/v6/study/manifest.json']
    files += [f'artifacts/v6/seed{s}/policy.pt' for s in (201, 211, 223)]
    result = {'version': 'v9', 'config': MergeConfig().dict(), 'radius': .5, 'model_seeds': [201, 211, 223],
              'test_seeds': {'nominal': seeds[:100], 'dense': seeds[100:]}, 'disturbances': ['clean', 'uniform', 'bias', 'pulse'],
              'methods': list(METHODS), 'executions': 5600, 'independent_initial_states': 200, 'retraining': False,
              'development': '24 previously published V8 scenes, four error modes and five methods (480 runs); all 200 V8 initial states checked for robust recoverability. Never counted as new evidence.',
              'uncertainty': 'Every 0.1 s, v+=clip(v+(u+w)*dt,0,vmax), |w|<=0.5 m/s^2; s+=v+*dt. Exact state measurements; slew limit concerns commanded u, not noisy actual acceleration.',
              'pairing': 'Same state and seed-indexed exogenous error tape across all methods; four modes repeat each initial state. Bootstrap clusters by seed, never treat 5600 executions as independent scenes.',
              'controls': 'V8 cached optimal plan; same-profile nominal feedback; robust constant; robust pair using V6 training-best index 1; random nonbaseline pair; frozen learned pair; robust full nominal search.',
              'criteria': {'robust_learned_pair_zero_collision_contract_violation_and_abort': True,
                           'robust_learned_pair_completion_fraction_min': .98,
                           'robust_learned_pair_perturbed_gain_vs_robust_constant_percent_min': 3.,
                           'robust_learned_pair_perturbed_paired_ci_lower_gt_zero': True,
                           'robust_learned_pair_failure_reduction_vs_v8_cached_min': .8,
                           'robust_learned_pair_warm_first_response_p95_ms_max': 20.},
              'limits': ['Known deterministic error bound, not empirical confidence or arbitrary model uncertainty.',
                         'Interval outer approximation with deterministic braking suffix; not a novel generic robust safety filter.',
                         'Only nominal finite-library performance comparison; no worst-case arrival-time dominance.',
                         'Fixed three-car cooperative routes; no perception noise, uncooperative agents, latency or real vehicle.'],
              'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    (D/'protocol.json').write_text(json.dumps(result, indent=2)+'\n', newline='\n')


if __name__ == '__main__': main()
