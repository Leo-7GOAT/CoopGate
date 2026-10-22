"""Freeze all single-agent checkpoints, thresholds and fresh tests before running."""
import hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v12/single'


def freeze():
    if (D/'protocol.json').exists():raise RuntimeError('Protocol already frozen')
    D.mkdir(parents=True,exist_ok=True)
    seeds=[411,421,431]
    training=ROOT/'runs/v12_single_release';calibration=ROOT/'runs/v12_single_calibration'
    required=[calibration/'calibration.json']+[training/f'ppo_{s}/metadata.json' for s in seeds]
    if not all(p.exists() for p in required):raise RuntimeError('Training/calibration incomplete')
    shutil.copytree(calibration,D/'calibration',dirs_exist_ok=False)
    shutil.copy2(training/'training_protocol.json',D/'training_protocol.json')
    for s in seeds:
        model_dir=D/f'seed_{s}';model_dir.mkdir()
        for src,dst in [(training/f'ppo_{s}/policy.pt','policy.pt'),(training/f'ppo_{s}/initial.pt','initial.pt'),
            (training/f'ppo_{s}/training.csv','training.csv'),(training/f'ppo_{s}/metadata.json','metadata.json'),
            (training/f'bc_{s}/bc.pt','bc.pt'),(training/f'bc_{s}/imitation.json','imitation.json')]:shutil.copy2(src,model_dir/dst)
    shutil.copytree(ROOT/'runs/v12_single_teacher',D/'teacher',dirs_exist_ok=False)
    shutil.copy2(ROOT/'runs/v12_single_twin_audit.json',D/'twin_audit.json')
    rng=np.random.default_rng(20261016123)
    test_seeds=rng.choice(np.arange(23000000,24000000),128,replace=False).tolist()
    initials=[{'seed':int(s),'regime':str(rng.choice(['nominal','dense']))} for s in test_seeds]
    methods=['rule','rollout']+[f'{mode}_{s}' for s in seeds for mode in ('bc','rl','gate')]
    source=[str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'src/coopgate').glob('*.py')]
    source+=['tools/study_single_v12.py','tools/experiment_single_v12.py','tools/calibrate_single_v12.py',
             'tools/report_single_v12.py','tools/freeze_single_v12.py']
    source+=['tools/train_v12.py','tools/teacher_v12.py','tools/teacher_single_v12.py',
             'tools/train_single_release_v12.py','tools/audit_twin_v12.py','tools/develop_v12.py']
    source += [str(p.relative_to(ROOT)).replace('\\','/') for p in D.rglob('*') if p.is_file()]
    protocol={'version':'v12-single','parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'training_seeds':seeds,'initials':initials,'disturbances':['clean','uniform','bias','pulse'],'methods':methods,
        'expected_runs':128*4*len(methods),'single_learning_vehicle':1,'objective_weight':.05,
        'policies':{str(s):f'artifacts/v12/single/seed_{s}/policy.pt' for s in seeds},
        'bc_policies':{str(s):f'artifacts/v12/single/seed_{s}/bc.pt' for s in seeds},
        'calibration':'artifacts/v12/single/calibration/calibration.json',
        'criteria':{'learned_cost_gain_vs_rollout_percent_min':3.,'learned_variation_gain_vs_rollout_percent_min':20.,
            'learned_arrival_regression_vs_rollout_percent_max':2.,'each_training_seed_cost_gain_min':0.,
            'all_rl_and_gate_completed':True,'no_rl_or_gate_safety_events':True,
            'gated_cost_gain_vs_rollout_percent_min':2.,'gate_retained_cost_gain_percent_min':50.,
            'harmful_acceptance_rate_max':.1,'gate_strictly_between_zero_and_full_acceptance':True},
        'bootstrap':{'seed':20261016124,'draws':10000,'cluster':'initial seed, aggregate four modes and three fixed trained policies',
            'interval':.95,'interpretation':'descriptive paired cluster intervals, not training-seed population inference'},
        'performance_gate_frequency':'once before each merge episode; safety verification every 0.5 seconds',
        'limits':['fixed-route three-car model','one trainable actor, not MARL','centralized shield can brake rule vehicles',
                  'conformal guarantee requires exchangeable calibration/test episodes within initial-state group',
                  'confidence is for regularized performance, not collision probability'],
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source}}
    (D/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'test_initials':len(initials),'sumo_runs':protocol['expected_runs'],'source_files':len(source)}))


if __name__=='__main__':freeze()
