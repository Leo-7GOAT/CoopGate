"""Freeze data separation, training/selection rules and test methods before use."""
import hashlib
import json
from pathlib import Path

import numpy as np

from coopgate.merge_v4 import MergeConfig


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v6'


def main():
    D.mkdir(exist_ok=True)
    if (D/'protocol.json').exists():
        raise RuntimeError('Refusing protocol overwrite')
    rng = np.random.default_rng(2026101507)
    seeds = rng.choice(np.arange(6000000, 7000000), size=1272, replace=False).tolist()
    files = ['src/coopgate/merge_v4.py', 'src/coopgate/planning_v5.py', 'src/coopgate/learning_v6.py',
             'tools/data_v6.py', 'tools/train_v6.py', 'tools/calibrate_v6.py', 'tools/study_v6.py']
    protocol = {'version': 'v6', 'config': MergeConfig().dict(),
                'scope': 'Centralized full-information contextual bandit selecting a fixed V5 feedback schedule; not decentralized MARL.',
                'data_seeds': {'train': seeds[:768], 'validation': seeds[768:896], 'calibration': seeds[896:1152]},
                'test_seeds': {'nominal': seeds[1152:1212], 'dense': seeds[1212:]},
                'data_distribution': 'Train, validation, calibration are nominal. Dense is uncalibrated distribution shift.',
                'training': {'seeds': [201, 211, 223], 'epochs': 180, 'warmup_epochs': 20,
                             'batch_size': 64, 'learning_rate': .001, 'teacher_temperature': .3,
                             'entropy_weight': .01, 'value_weight': .5, 'validation_interval': 10,
                             'selection': 'Lowest validation greedy arrival cost from epoch 20 onward; earliest checkpoint wins ties.'},
                'calibration_alpha': .1,
                'calibration_rule': 'q is ceil((n+1)*(1-alpha))-th residual order statistic. Accept when predicted selected excess + q <= 0, else use profile zero.',
                'ensemble': 'Average action probabilities and excess-cost predictions of all three validation-selected models; no ensemble-member selection.',
                'methods': ['constant', 'fixed', 'imitation', 'learned', 'calibrated', 'teacher'],
                'executions': 720,
                'fixed_baseline': 'One profile minimizing mean train-set cost, without test access.',
                'limits': ['Performance bound is marginal over calibration and new nominal scenes, not conditional on acceptance.',
                           'It does not bound dense/OOD errors, collision risk, or repeated online decisions.',
                           'Safety remains the separate exact-model joint braking contract.',
                           'All 37 counterfactual costs are model-generated training labels; no online trial-and-error learning claim.'],
                'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
    old = json.loads((ROOT/'artifacts/v5/protocol.json').read_text())
    assert not set(seeds).intersection(s['seed'] for scenes in old['splits'].values() for s in scenes)
    (D/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', newline='\n')
    print(D/'protocol.json')


if __name__ == '__main__':
    main()
