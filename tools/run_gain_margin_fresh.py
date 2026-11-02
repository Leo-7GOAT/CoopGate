"""Four owned native collectors/reviewers;complete fixed-family development only."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from evaluate_gain_margin_fresh import schedule,summarize,METHODS,prerequisite,sha,write

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'


def run():
    dependency=prerequisite();execution=BASE/'gain_margin_fresh_20261102_execution';execution.mkdir(exist_ok=False)
    paths=[BASE/f'development/gain_margin_fresh_20261102_shard{i}' for i in range(4)]
    write(execution/'plan.json',dict(shards=[str(p.relative_to(ROOT)) for p in paths],prerequisite=dependency,
        expected_sumo_executions=1040,independent_initials=80,error_allocation=None,risk_certificate=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    stages=[]
    for action in ('collect','review'):
        running=[]
        for shard,out in enumerate(paths):
            command=[sys.executable,str(ROOT/'tools/evaluate_gain_margin_fresh.py'),action,'--out',str(out),'--shard',str(shard)]
            log=(execution/f'{action}_{shard}.log').open('x',encoding='utf-8')
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            event=dict(stage=action,shard=shard,pid=child.pid,command=command,started_utc=datetime.now(timezone.utc).isoformat())
            with (execution/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            running.append((child,log,event))
        failed=False
        for child,log,event in running:
            code=child.wait();log.close();event.update(actual_native_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat())
            stages.append(event);failed|=code!=0
            with (execution/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        if failed:raise SystemExit(1)
    records=[];sources={};commands=substeps=0
    for shard,path in enumerate(paths):
        proof=json.loads((path/'verification.json').read_text());complete=json.loads((path/'collection_complete.json').read_text())
        if not proof['all_match'] or sha(path/'episodes.jsonl')!=complete['episodes_sha256']:raise ValueError('Verified unchanged shard required')
        records.extend(json.loads(s) for s in (path/'episodes.jsonl').read_text().splitlines())
        sources[path.name]=dict(episodes_sha256=sha(path/'episodes.jsonl'),verification_sha256=sha(path/'verification.json'))
        commands+=proof['commands_matched'];substeps+=proof['independent_physical_substeps']
        write(path/'native_receipt.json',dict(collect_actual_native_exit_code=0,review_actual_native_exit_code=0,
            stages=[e for e in stages if e['shard']==shard]))
    cases=[case for shard in range(4) for case in schedule(shard)];lookup={tuple(r['key']):r for r in records}
    expected={tuple(c)+(method,) for c in cases for method in METHODS}
    if len(records)!=1040 or len(lookup)!=1040 or set(lookup)!=expected or len({c[0] for c in cases})!=80:
        raise ValueError('Complete1040-outcome/80-distinct-initial family required')
    zero_adopt_exact=0;prefix_coupled=True
    for r in records:
        if r['row']['method_id']=='baseline':continue
        baseline=lookup[tuple(r['key'][:4])+('baseline',)]
        for item in r['trace']:
            if item['actual_learning_used']:break
            prefix_coupled&=np.asarray(item['command']).tobytes()==np.asarray(item['baseline_command']).tobytes()
        if r['row']['learned_action_steps']==0 and r['row']['completed']:
            if len(r['trace'])!=len(baseline['trace']):raise ValueError('Successful zero-adoption baseline trajectory length differs')
            for a,b in zip(r['trace'],baseline['trace']):
                if np.asarray(a['command']).tobytes()!=np.asarray(b['command']).tobytes() or np.asarray(a['x']).tobytes()!=np.asarray(b['x']).tobytes():
                    raise ValueError('Successful zero-adoption trajectory is not exactly the paired baseline')
            zero_adopt_exact+=1
    nested=True
    for case in cases:
        for seed in (4703,4704,4705):
            adopted=[lookup[tuple(case)+(f'gain_{seed}_{floor}',)]['row']['learned_action_steps']>0 for floor in (0,25,50,100)]
            nested&=all(not adopted[i+1] or adopted[i] for i in range(3))
    if prefix_coupled and not nested:raise ValueError('Nested score floors violate episode-adoption coupling invariant')
    report=dict(methods=summarize(records,cases),actual_sumo_executions=1040,independent_initials=80,
        commands_verified=commands,physical_substeps_verified=substeps,sources=sources,
        successful_zero_adopt_exact_baseline_runs=zero_adopt_exact,exact_baseline_commands_before_first_adoption=bool(prefix_coupled),
        observed_episode_adoption_nested_by_floor=bool(nested),
        formal_calibration=False,error_allocation=None,risk_certificate=False,paper_ready=False,
        inference='Balanced factorial DEVELOPMENT,not IID full mixture. All methods/floors retained;no certification or independent-sample inflation.')
    write(BASE/'gain_margin_fresh_development_20261102.json',report)
    write(execution/'completion.json',dict(stages=stages,all_actual_native_exits_zero=True,
        report_sha256=sha(BASE/'gain_margin_fresh_development_20261102.json'),actual_sumo_executions=1040,independent_initials=80,risk_certificate=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('methods','sources')}),flush=True)


if __name__=='__main__':run()
