"""Freeze V10 sources, held-out seeds and meaningful-improvement criteria."""
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.merge_v4 import MergeConfig
from coopgate.experiment_v10 import METHODS

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v10'


def main():
    D.mkdir(exist_ok=True)
    if (D/'protocol.json').exists(): raise RuntimeError('V10 protocol already frozen')
    seeds=np.random.default_rng(2026101610).choice(np.arange(15000000,16000000),200,replace=False).tolist()
    files=['src/coopgate/rollout_v10.py','src/coopgate/experiment_v10.py','tools/study_v10.py',
           'src/coopgate/robust_v9.py','src/coopgate/disturbance_v9.py','src/coopgate/fast_v8.py',
           'src/coopgate/merge_v4.py','src/coopgate/planning_v5.py','src/coopgate/learning_v6.py',
           'tests/test_rollout_v10.py','artifacts/v6/study/manifest.json']
    files += [f'artifacts/v6/seed{s}/policy.pt' for s in (201,211,223)]
    development={}
    for name in ('v10_development','v10_development_adaptive'):
        raw=(ROOT/f'runs/{name}.json').read_bytes(); rows=json.loads(raw)
        methods=list(dict.fromkeys(r['method'] for r in rows))
        development[name]={'executions':len(rows),'old_initial_states':len({r['seed'] for r in rows}),
            'raw_sha256':hashlib.sha256(raw).hexdigest(),
            'mean_cost_by_error':{mode:{m:float(np.mean([r['mean_capped_finish_s'] for r in rows if r['disturbance']==mode and r['method']==m])) for m in methods} for mode in ('clean','uniform','bias','pulse')}}
    development['selection']='Single adaptive coordinate sweep, five levels. Two sweeps had no clean/uniform gain and nearly doubled late-cycle work. Learning adds little in development; retain no-learning, fixed/random and binary controls without promising a learning benefit.'
    (D/'development.json').write_text(json.dumps(development,indent=2)+'\n',newline='\n')
    protocol={'version':'v10','config':MergeConfig().dict(),'radius':.5,'model_seeds':[201,211,223],
        'test_seeds':{'nominal':seeds[:100],'dense':seeds[100:]},
        'disturbances':['clean','uniform','bias','pulse'],'methods':list(METHODS),
        'executions':6400,'independent_initial_states':200,'retraining':False,
        'algorithm':{'levels':5,'coordinate_passes':1,'adaptive_base_selection':True,
                     'strict_integer_cost_improvement':True,'global_profile_clock':True,
                     'pruning':'independent earliest arrival lower bound; retain incumbent on ties'},
        'development':'16 published V9 states / 320 runs, then 32 published V9 states / 896 runs; reused old states are development only.',
        'pairing':'Same initial state and exogenous V9 disturbance tape across all methods. Randomized method order. Three perturbed conditions form one seed cluster for bootstrap.',
        'primary_contrast':'rollout_learned versus v9_learned_pair over three nonzero error modes',
        'criteria':{'zero_collision_contract_violation_and_abort':True,
                    'completion_fraction_min':.99,
                    'perturbed_gain_vs_v9_learned_percent_min':3.,
                    'primary_paired_cluster_ci_lower_gt_zero':True,
                    'perturbed_gain_vs_v9_full_percent_min':2.,
                    'perturbed_dense_gain_vs_v9_learned_percent_min':2.,
                    'clean_worse_than_v9_learned_count_max':0,
                    'warm_controller_cycle_p95_ms_max':20.,
                    'warm_first_response_p95_ms_max':25.},
        'inference':'10000 paired seed-cluster bootstrap draws, RNG 2026101611, descriptive 95% intervals. Secondary comparisons do not receive familywise significance claims.',
        'limits':['Known error bound and exact measurements; fixed three-car cooperative routes.',
                  'Nominal policy-improvement theorem only; no realized performance dominance under disturbances.',
                  'Learning is the frozen centralized V6 contextual bandit; no new MARL training.',
                  'Coordinate rollout and predictive safety filters are prior art; domain composition and evidence are the contribution.',
                  'SUMO pauses during computation; warm measurements are not worst-case real-time or communication-delay guarantees.'],
        'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    (D/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',newline='\n')


if __name__=='__main__':main()
