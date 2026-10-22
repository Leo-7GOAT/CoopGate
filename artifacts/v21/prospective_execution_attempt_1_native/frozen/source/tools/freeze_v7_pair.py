"""Freeze a separate new-seed study following calibration-only diagnosis."""
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.merge_v4 import MergeConfig

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v7_pair'


def main():
    D.mkdir(exist_ok=True)
    if (D/'protocol.json').exists():
        raise RuntimeError('Already frozen')
    seeds = np.random.default_rng(2026101518).choice(np.arange(9000000, 10000000), 80, replace=False).tolist()
    sources = ['src/coopgate/verify_v7.py', 'src/coopgate/learning_v6.py', 'src/coopgate/planning_v5.py',
               'src/coopgate/merge_v4.py', 'tools/study_v7_pair.py', 'artifacts/v7/calibration.json']
    sources += [f'artifacts/v6/seed{s}/policy.pt' for s in (201, 211, 223)]
    protocol = {'version': 'v7_pair', 'config': MergeConfig().dict(), 'model_seeds': [201, 211, 223],
                'test_seeds': {'nominal': seeds[:40], 'dense': seeds[40:]},
                'methods': ['constant', 'learned', 'verified_pair', 'teacher'], 'executions': 320,
                'motivation': 'V7 calibration alone found no simultaneously certified threshold; independently evaluate online two-profile verification without inspecting V7 test results.',
                'rule': 'Compare full-episode model costs of baseline and learned proposal; choose lower, baseline wins ties. Execute cached plan with existing certificate and invalidation fallback.',
                'limits': 'Exact model relative performance only; after plan invalidation no original performance guarantee. Not a new statistical gate, no bounded-deadline claim.',
                'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}}
    (D/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', newline='\n')


if __name__ == '__main__':
    main()
