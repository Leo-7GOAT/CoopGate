"""Development only; old published scenes, never held-out V9 evidence."""
import json
from pathlib import Path
import time
import numpy as np
from coopgate.merge_v4 import MergeConfig
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.robust_v9 import RobustGate
from coopgate.fast_v8 import plan_fast, FastPlannedGate
from coopgate.learning_v6 import load_models, predict

ROOT = Path(__file__).resolve().parents[1]
models = load_models(ROOT/'artifacts/v6', [201, 211, 223])
old = json.loads((ROOT/'artifacts/v8/protocol.json').read_text())
rows = []
with DisturbedMergeEnv(MergeConfig(), 'constant') as env:
    for regime, seeds in old['test_seeds'].items():
        for mode in ('clean', 'uniform', 'bias', 'pulse'):
            for seed in seeds[:12]:
                for method in ('v8_cached', 'nominal_feedback', 'robust_constant', 'robust_pair', 'robust_full'):
                    env.reset_disturbed(seed, regime, mode, .5); start = time.perf_counter()
                    if method == 'v8_cached':
                        plan, _ = plan_fast(env.x, env.prev, env.c, pruning=True); env.gate = FastPlannedGate(env.c, plan)
                    else:
                        p = int(predict(models, env.x[None])[0][0]) if method == 'robust_pair' else 0
                        strategy = 'constant' if method == 'robust_constant' else 'pair' if method == 'robust_pair' else 'full'
                        env.gate = RobustGate(env.x, env.prev, env.c, radius=0. if method == 'nominal_feedback' else .5, strategy=strategy, proposal=p)
                    planning_ms = (time.perf_counter()-start)*1000
                    while True:
                        _, _, done, truncated, metrics = env.step([4]*3)
                        if done or truncated: break
                    rows.append({**metrics, 'method': method, 'planning_ms': planning_ms})
            print(json.dumps({'scenario':regime,'mode':mode,'rows':len(rows)}), flush=True)
(ROOT/'runs/v9_development.json').write_text(json.dumps(rows, indent=2))
for mode in ('clean', 'uniform', 'bias', 'pulse'):
    for method in ('v8_cached', 'nominal_feedback', 'robust_constant', 'robust_pair', 'robust_full'):
        group = [r for r in rows if r['disturbance']==mode and r['method']==method]
        print(mode, method, 'cost',round(np.mean([r['mean_capped_finish_s'] for r in group]),3),
              'completed',sum(r['completed'] for r in group),'violation',sum(r['contract_violations'] for r in group),
              'aborts',sum(r['aborted'] for r in group),'planP95',round(np.percentile([r['planning_ms'] for r in group],95),2))
