"""Native-exit-observed fixed single-PPO replication; no retries."""
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


def run(seed):
    if seed not in (4602,4603):
        raise ValueError('Only the two prospectively registered replications')
    steps=262144;initial_seed={4602:95800001,4603:95900001}[seed]
    parent=BASE/'legacy_901/initializer.pt'
    first=json.loads((BASE/'single_4601_sumo_full/single_competence.json').read_text())
    if not first['numeric_thresholds_met']:
        raise ValueError('Original complete I competence assessment required')
    directory=ROOT/f'artifacts/mappo_ctde_20261101/single_{seed}_execution'
    directory.mkdir(parents=True,exist_ok=False)
    names=('train_mappo_ctde.py','audit_mappo_ctde.py','evaluate_mappo_ctde.py',
           'verify_mappo_sumo.py','assess_mappo_single.py','run_single_mappo_replication.py')
    files=list((ROOT/'src/coopgate').glob('*.py'))+[ROOT/'tools'/n for n in names]
    files+=[ROOT/'docs/mappo_ctde_20261101_protocol.md',ROOT/'docs/mappo_replication_20261101_protocol.md',parent]
    frozen={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    plan=dict(seed=seed,steps=steps,initial_seed=initial_seed,source_sha256=frozen,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        expected_exact_executions=2560,conditional_sumo_executions=2560,
        no_retries=True,no_checkpoint_selection=True,new_risk_certificate=False,
        runtime='Concurrent CPU pipelines allowed; not isolated timing evidence')
    (directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    events=[]

    def record(value):
        value['utc']=datetime.now(timezone.utc).isoformat();events.append(value)
        with (directory/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)

    def command(name,args):
        for path,expected in frozen.items():
            if sha(ROOT/path)!=expected:raise ValueError('Frozen dependency changed: '+path)
        path=directory/(name+'.log');started=time.perf_counter()
        with path.open('x',encoding='utf-8') as output:
            child=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=output,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            record(dict(stage=name,event='started',native_pid=child.pid,argv=[sys.executable,*args]))
            code=child.wait()
        record(dict(stage=name,event='exited',actual_native_exit_code=code,
                    elapsed_s=time.perf_counter()-started,log_sha256=sha(path)))
        if code!=0:raise RuntimeError(f'{name} exited {code}; prefix preserved, no retry')

    own=BASE/f'single_{seed}';exact=BASE/f'single_{seed}_exact'
    method=f'single_central_{seed}'
    try:
        command('train',['tools/train_mappo_ctde.py','--out',str(own),'--steps',str(steps),'--seed',str(seed),
            '--stage','single','--width','256','--kinematic','--role-actors','--actor-relu',
            '--logit-scale','5','--entropy','.001','--gae-lambda','1','--actor-lr','.00001',
            '--legacy-initial',str(parent)])
        command('training-audit',['tools/audit_mappo_ctde.py',str(own)])
        common=['--paths',str(own/'policy.pt'),'--legacy-initial',str(parent),
                '--n','128','--seed-base',str(initial_seed)]
        command('exact',['tools/evaluate_mappo_ctde.py','--out',str(exact),*common,'--backend','exact'])
        command('exact-competence',['tools/assess_mappo_single.py',str(exact),'--method',method])
        result=json.loads((exact/'single_competence.json').read_text())
        eligible=bool(result['numeric_thresholds_met'])
        if eligible:
            full=BASE/f'single_{seed}_sumo_full'
            command('sumo-full',['tools/evaluate_mappo_ctde.py','--out',str(full),*common,'--backend','sumo'])
            command('sumo-full-review',['tools/verify_mappo_sumo.py',str(full)])
            command('sumo-competence',['tools/assess_mappo_single.py',str(full),'--method',method])
        else:
            record(dict(stage='sumo-full',event='not-enabled-by-preregistered-development-condition',
                        exact_competence=result))
        completion=dict(status='Fixed replication finished; scientific assessment separate',
            native_commands=[e for e in events if e['event']=='exited'],
            exact_numeric_thresholds_met=eligible,policy_sha256=sha(own/'policy.pt'),goal_complete=False)
        (directory/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    except BaseException as exc:
        record(dict(event='stopped',error_type=type(exc).__name__,error=str(exc),goal_complete=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    run(parser.parse_args().seed)
