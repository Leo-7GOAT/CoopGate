"""Freeze final models and decision rules before fresh SUMO tests."""
import hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v13/ego'


def freeze():
    if D.exists():raise RuntimeError('V13 ego release already exists')
    training=ROOT/'runs/v13_ego_release';calibration=ROOT/'runs/v13_ego_calibration'
    if not (calibration/'calibration.json').exists():raise RuntimeError('Calibration incomplete')
    train=json.loads((training/'training_protocol.json').read_text())
    for path,digest in train['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    D.mkdir(parents=True);shutil.copy2(training/'training_protocol.json',D/'training_protocol.json')
    shutil.copytree(calibration,D/'calibration')
    seeds=train['training_seeds']
    for s in seeds:
        dst=D/f'seed_{s}';dst.mkdir()
        for name in ('policy.pt','initial.pt','metadata.json','training.csv'):shutil.copy2(training/f'dqn_{s}'/name,dst/name)
    rng=np.random.default_rng(202610163213)
    initials=[{'seed':int(s),'regime':str(rng.choice(['nominal','dense']))} for s in rng.choice(np.arange(32000000,33000000),128,replace=False)]
    methods=['rule','rollout']+[f'{mode}_{s}' for s in seeds for mode in ('raw','untrained','augmented','gate')]
    source=[str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'src/coopgate').glob('*.py')]
    source+=['tools/train_dqn_ego_v13.py','tools/freeze_ego_training_v13.py','tools/calibrate_ego_v13.py',
        'tools/experiment_ego_v13.py','tools/freeze_ego_v13.py','tools/study_ego_v13.py','tools/report_ego_v13.py']
    source += [str(p.relative_to(ROOT)).replace('\\','/') for p in D.rglob('*') if p.is_file()]
    protocol={'version':'v13-ego','parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'initials':initials,'disturbances':['clean','uniform','bias','pulse'],'methods':methods,'training_seeds':seeds,
        'expected_runs':128*4*len(methods),'cost':'ego arrival + 0.05 own command variation until ego arrival',
        'alpha':.1,'cost_margin':.05,'time_margin':0.,'performance_gate':'once before each merge',
        'criteria':{'augmented_cost_gain_percent_min':1.,'augmented_variation_gain_percent_min':10.,
            'augmented_time_gain_percent_min':.5,'every_seed_augmented_cost_improves':True,
            'augmented_gain_vs_untrained_percent_min':1.,'gated_cost_gain_percent_min':.5,
            'gate_retains_cost_gain_percent_min':50.,'gated_time_regression_percent_max':0.,
            'joint_harmful_admission_rate_max':.1,'completed_and_no_safety_events':True,
            'real_admissions_and_fallbacks':True},
        'bootstrap':{'seed':202610163214,'draws':10000,'cluster':'initial; average four modes and three fixed trained models','interval':.95},
        'interpretation':['standalone actor and learning-augmented rollout reported separately',
            'no imitation data; three fresh fixed final Double DQN checkpoints',
            'does not establish standalone RL dominance or justify MARL expansion unless standalone criteria also pass',
            'timing is synchronous simulation, not a delay-aware hardware test'],
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source}}
    (D/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'runs':protocol['expected_runs'],'files':len(source),'criteria':protocol['criteria']}))


if __name__=='__main__':freeze()
