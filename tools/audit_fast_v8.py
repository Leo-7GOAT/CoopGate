"""Development audit using already-published data, never fresh test outcomes."""
import json
from pathlib import Path
import time
import numpy as np
from coopgate.fast_v8 import learned_order, plan_fast
from coopgate.learning_v6 import load_models
from coopgate.merge_v4 import MergeConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    models = load_models(ROOT/'artifacts/v6', [201, 211, 223]); c = MergeConfig(); records = []
    start = time.perf_counter()
    plan_fast(np.array([[-80., 12.], [-85., 13.], [-120., 11.]]), np.zeros(3), c)
    startup = (time.perf_counter()-start)*1000
    for version in ('v6', 'v7', 'v7_pair'):
        labels = json.loads((ROOT/f'artifacts/{version}/study/teacher_labels.json').read_text())
        for label in labels:
            x = np.array(label['initial']); expected = np.array(label['costs']); ordering = learned_order(models, x)
            row = {'version': version, 'seed': label['seed'], 'scenario': label['scenario']}
            for method, prune, order in [('full', False, None), ('natural', True, None), ('learned', True, ordering)]:
                plan, stats = plan_fast(x, np.zeros(3), c, order=order, pruning=prune)
                if method == 'full':
                    np.testing.assert_array_equal(plan.costs, expected)
                assert plan.selected == np.argmin(expected), (method, label['seed'], plan.selected, expected)
                assert plan.predicted_cost == expected.min()
                row[method] = {'ms': plan.planning_ms, **stats}
            records.append(row)
        print(json.dumps({'audited': len(records)}), flush=True)
    output = {'first_call_ms': startup, 'records': records,
              'aggregate': {m: {'p95_ms': float(np.percentile([r[m]['ms'] for r in records], 95)),
                                'mean_checks': float(np.mean([r[m]['certificate_calls'] for r in records]))}
                            for m in ('full', 'natural', 'learned')}}
    path = ROOT/'runs/v8_development_audit.json'; path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output['aggregate'], indent=2))


if __name__ == '__main__':
    main()
