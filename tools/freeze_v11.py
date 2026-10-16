"""Create an immutable new study protocol before running fresh SUMO scenes."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from coopgate.experiment_v11 import METHODS

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v11'


def main():
    if (D/'protocol.json').exists():
        raise FileExistsError('V11 protocol already frozen')
    D.mkdir(parents=True,exist_ok=True)
    (D/'development').mkdir(exist_ok=False)
    development={}
    for name in ('v11_development','v11_development_expanded','v11_development_families','v11_development_ordered'):
        source=ROOT/f'runs/{name}.json';target=D/f'development/{name}.json'
        shutil.copyfile(source,target);rows=json.loads(source.read_text())
        development[name]={'executions':len(rows),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'methods':{m:{'perturbed_count':len(g),'mean_cost_s':float(np.mean([r['mean_capped_finish_s'] for r in g])),
                'mean_total_variation':float(np.mean([r['total_variation'] for r in g]))}
                for m in dict.fromkeys(r['method'] for r in rows)
                for g in [[r for r in rows if r['method']==m and r['disturbance']!='clean']]}}
    development['notes']=[
        'All development uses published V10 seeds, never the new V11 test seeds.',
        'Initial phase: first 8 seeds per regime, 320 runs; lex_ternary there means hold-only tails.',
        'Expanded phase: first 24 seeds per regime, 1152 runs; lex_ternary retains both families but improves only the best raw tail family.',
        'Family sweeps: 192 runs on the same 48 states, both families each receive a sweep.',
        'Ordered projection: 192 repeated runs, same episode results as family sweeps. Development timings are not formal single-worker benchmarks.',
        'Integration smoke: 72 SUMO runs on the first published V10 seed per regime, all completed; retained in local runs/v11_study_smoke.',
        'Chosen complete method: .625 command lattice, lexicographic objective, both families, one coordinate sweep per family.']
    (D/'development.json').write_text(json.dumps(development,indent=2)+'\n',newline='\n')
    old=json.loads((ROOT/'artifacts/v10/protocol.json').read_text())
    seeds=np.random.default_rng(2026101612).choice(np.arange(17000000,18000000),200,replace=False).tolist()
    sources=list(old['source_sha256'])+['src/coopgate/smooth_v11.py','src/coopgate/experiment_v11.py',
        'tools/study_v11.py','tools/report_v11.py','tests/test_smooth_v11.py','tools/freeze_v11.py',
        'docs/theory_v11.md','tools/replay_v11.py']
    protocol={'version':'v11','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':old['config'],'radius':.5,'model_seeds':[201,211,223],
        'test_seeds':{'nominal':seeds[:100],'dense':seeds[100:]},
        'disturbances':['clean','uniform','bias','pulse'],'methods':list(METHODS),
        'executions':7200,'independent_initial_states':200,'retraining':False,
        'algorithm':{'command_quantum':.625,'coordinate_passes_per_family':1,'families':['binary','hold'],
            'profile_selection':'Best raw full-tail lexicographic value separately within each family',
            'final_selection':'Best of global raw incumbent and the two improved family candidates',
            'objectives':['integer unfinished-vehicle ticks','integer command total-variation units'],
            'global_profile_clock':True,'pruning':'lexicographic (independent earliest arrival ticks, accumulated variation) lower bound',
            'projection':'stable score order; first jointly certified finite vertex'},
        'primary_contrast':'smooth_learned versus frozen v10_rollout, averaged over the three nonzero error modes per initial state',
        'co_primary_endpoints':['mean_capped_finish_s','total_command_variation'],
        'criteria':{'zero_collision_contract_violation_and_abort':True,'completion_fraction_min':1.,
            'perturbed_time_gain_vs_v10_percent_min':1.,'perturbed_variation_reduction_vs_v10_percent_min':25.,
            'time_paired_cluster_ci97_5_lower_gt_zero':True,'variation_paired_cluster_ci97_5_lower_gt_zero':True,
            'perturbed_jerk_reduction_vs_v10_percent_min':25.,'clean_worse_than_v9_count_max':0,
            'warm_controller_cycle_p95_ms_max':20.,'warm_first_response_p95_ms_max':35.},
        'bootstrap_seed':2026101613,'bootstrap_draws':10000,
        'inference':'Two prespecified endpoints use 97.5% paired seed-cluster percentile bootstrap intervals (Bonferroni allocation); approximate, not a distribution-free guarantee. Secondary contrasts descriptive.',
        'pairing':'Identical actual SUMO initial states and frozen V9 exogenous error tapes. Method order shuffled with seed+1103+error-mode-index. Single worker, prewarmed.',
        'limits':['Nominal lexicographic guarantee is not componentwise dominance and is not dominance over V10.',
            'Physical certificate unchanged; no performance guarantee under actual disturbances.',
            'Frozen V6 centralized contextual bandit, no new MARL training.',
            'Three cooperative fixed-route cars, known actuator-error bound, exact measurements.',
            'Command total variation and change rate are not measured physical jerk or passenger comfort.',
            'SUMO pauses during controller computation; no worst-case timing or communication-delay guarantee.'],
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in dict.fromkeys(sources)}}
    with (D/'protocol.json').open('x',newline='\n') as f:f.write(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'seeds':200,'executions':protocol['executions'],'source_files':len(protocol['source_sha256']),
        'criteria':protocol['criteria']},indent=2))


if __name__=='__main__':main()
