"""Frozen, native-exit-observed joint MAPPO development pipeline.

No retries or checkpoint selection. Scientific negative results are retained.
The parent owns each child handle and records its actual native return code.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101/development'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def run(seed=4701,steps=1048576):
    settings={4701:dict(steps=1048576,initial_seed=95500001,protocol='mappo_joint_20261101_protocol.md',rear_coast=False),
              4702:dict(steps=2097152,initial_seed=95600001,protocol='mappo_joint_coast_20261101_protocol.md',rear_coast=True),
              4703:dict(steps=2097152,initial_seed=95700001,protocol='mappo_joint_ego_stable_20261101_protocol.md',rear_coast=True,ego_actor_lr=1e-5,role_reference=True),
              4704:dict(steps=2097152,initial_seed=96000001,protocol='mappo_joint_replications_20261101_protocol.md',rear_coast=True,ego_actor_lr=1e-5,role_reference=True),
              4705:dict(steps=2097152,initial_seed=96100001,protocol='mappo_joint_replications_20261101_protocol.md',rear_coast=True,ego_actor_lr=1e-5,role_reference=True)}
    if seed not in settings or steps!=settings[seed]['steps']:
        raise ValueError('Use a registered fixed joint-stage seed/budget')
    config=settings[seed]
    study=f'joint_{seed}'
    parent=BASE/'single_4601/policy.pt'
    proof=BASE/'single_4601_sumo_full'
    report=json.loads((proof/'report.json').read_text())
    verification=json.loads((proof/'verification.json').read_text())
    competence=json.loads((proof/'single_competence.json').read_text())
    result=report['results']['single_central_4601']
    if (report['protocol']['backend']!='sumo' or report['protocol']['n_independent_initials']!=128
            or result['failures']!=0 or result['comparisons']['ego_planner']['per_vehicle_gain_one_sided_t_lower_s'][1]<=0
            or not verification['all_match'] or verification['actual_sumo_episodes']!=2560
            or not competence['numeric_thresholds_met']):
        raise ValueError('Full new single-PPO SUMO superiority and complete replay required')
    execution=ROOT/f'artifacts/mappo_ctde_20261101/{study}_execution'
    execution.mkdir(parents=True,exist_ok=False)
    names=['train_mappo_ctde.py','evaluate_mappo_ctde.py','verify_mappo_sumo.py',
           'audit_mappo_ctde.py','analyze_mappo_development.py','archive_mappo_ctde.py','run_joint_mappo_stage.py']
    paths=list((ROOT/'src/coopgate').glob('*.py'))+[ROOT/'tools'/n for n in names]
    paths += [ROOT/'docs/mappo_ctde_20261101_protocol.md',ROOT/'docs'/config['protocol'],
              parent,proof/'report.json',proof/'verification.json',proof/'single_competence.json']
    frozen={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    plan=dict(seed=seed,steps=steps,created_utc=utc(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              frozen_sha256=frozen,condition='New single PPO full SUMO ramp superiority and replay passed',
              actual_single_proof_sha256=sha(proof/'verification.json'),
              primary_initials=128,primary_seed_base=config['initial_seed'],modes=4,
              rear_coast_initialization=config['rear_coast'],
              ego_actor_lr=config.get('ego_actor_lr',1e-4),
              exact_expected_executions=512*(10 if config.get('role_reference') else 9),
              role_replacement='All three roles separately restored to original parameters, full paired closed-loop evaluation; no new independent initials',
              sumo_integration_expected_executions=8*(10 if config.get('role_reference') else 9),
              full_sumo_condition='Zero failure, positive raw joint mean gain and positive development one-sided t lower vs strong team planner',
              full_sumo_expected_executions_if_condition_met=3072,
              no_retries=True,no_checkpoint_selection=True)
    (execution/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    events=[]

    def record(value):
        events.append(value)
        with (execution/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)

    def check_sources():
        for name,expected in frozen.items():
            if sha(ROOT/name)!=expected:
                raise ValueError('Frozen joint-study dependency changed: '+name)

    def command(name,args):
        check_sources()
        logfile=execution/(name+'.log')
        started=time.perf_counter()
        with logfile.open('x',encoding='utf-8') as output:
            child=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=output,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            record(dict(stage=name,event='started',utc=utc(),native_pid=child.pid,argv=[sys.executable,*args]))
            code=child.wait()
        record(dict(stage=name,event='exited',utc=utc(),actual_native_exit_code=code,
                    elapsed_s=time.perf_counter()-started,log_sha256=sha(logfile)))
        if code!=0:
            raise RuntimeError(f'{name} exited {code}; prefix preserved, no retry')

    own=BASE/study
    exact=BASE/f'{study}_exact'
    integration=BASE/f'{study}_sumo_integration'
    common=['--paths',str(parent),str(own/'policy.pt'),'--joint-initial',str(own/'initial.pt')]
    role_reference_args=['--role-reference',str(parent)] if config.get('role_reference') else []
    try:
        train_args=['tools/train_mappo_ctde.py','--out',str(own),'--steps',str(steps),'--seed',str(seed),
            '--stage','joint','--width','256','--kinematic','--role-actors','--actor-relu',
            '--logit-scale','5','--entropy','.01','--gae-lambda','1','--actor-lr','.0001','--initial',str(parent)]
        if config['rear_coast']:
            train_args.append('--rear-coast')
        if 'ego_actor_lr' in config:
            train_args.extend(['--ego-actor-lr',str(config['ego_actor_lr'])])
        command('train',train_args)
        command('training-audit',['tools/audit_mappo_ctde.py',str(own)])
        command('exact',['tools/evaluate_mappo_ctde.py','--out',str(exact),*common,
            '--revert-final',str(own/'policy.pt'),*role_reference_args,'--n','128','--seed-base',str(config['initial_seed']),'--backend','exact'])
        command('sumo-integration',['tools/evaluate_mappo_ctde.py','--out',str(integration),*common,
            '--revert-final',str(own/'policy.pt'),*role_reference_args,'--n','2','--seed-base',str(config['initial_seed']),'--backend','sumo'])
        command('sumo-integration-review',['tools/verify_mappo_sumo.py',str(integration)])
        command('conflict-analysis',['tools/analyze_mappo_development.py',str(exact)])
        primary=json.loads((exact/'report.json').read_text())
        joint=primary['results'][f'joint_central_{seed}']
        contrast=joint['comparisons']['team_planner']
        eligible=bool(joint['failures']==0 and contrast['all_scope_gain_assessable']
                      and contrast['mean_team_gain_s']>0 and contrast['team_gain_one_sided_t_lower_s']>0)
        if eligible:
            full=BASE/f'{study}_sumo_full'
            command('sumo-full',['tools/evaluate_mappo_ctde.py','--out',str(full),*common,
                '--n','128','--seed-base',str(config['initial_seed']),'--backend','sumo'])
            command('sumo-full-review',['tools/verify_mappo_sumo.py',str(full)])
        else:
            record(dict(stage='sumo-full',event='not-enabled-by-preregistered-development-condition',
                        raw_joint_vs_team=contrast,failures=joint['failures']))
        completion=dict(status='registered development stages completed; scientific verdict is separate',
            native_commands=[e for e in events if e['event']=='exited'],
            raw_joint_eligible_for_full_sumo=eligible,policy_sha256=sha(own/'policy.pt'),
            exact_report_sha256=sha(exact/'report.json'),goal_complete=False)
        (execution/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    except BaseException as exc:
        record(dict(event='stopped',utc=utc(),error_type=type(exc).__name__,error=str(exc),goal_complete=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=4701)
    parser.add_argument('--steps',type=int,default=1048576)
    args=parser.parse_args();run(args.seed,args.steps)
