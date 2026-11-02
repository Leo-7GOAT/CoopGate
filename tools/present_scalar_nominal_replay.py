"""Independently reaggregate all paired callbacks and render a scientific figure."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    source=BASE/'development/scalar_nominal_full_20261102'
    out=BASE/'scalar_nominal_replay_analysis_20261102';out.mkdir(exist_ok=False)
    report=json.loads((source/'report.json').read_text());native=json.loads((source/'native_receipt.json').read_text())
    plan=json.loads((source/'plan.json').read_text())
    if native['actual_native_exit_code']!=0 or sha(source/'callbacks.jsonl')!=report['callbacks_sha256']:
        raise ValueError('Observed terminal zero and unmodified complete replay required')
    source_rows={}
    for i in range(4):
        p=BASE/f'development/parameter_scenario_grid_20261102_shard{i}/episodes.jsonl'
        if sha(p)!=plan['source_hashes'][p.parent.name]:raise ValueError('Original SUMO source changed')
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if r['row']['rule']=='parameter_scenario_confidence':source_rows[tuple(r['key'])]=r
    expected={(key,name,step) for key,r in source_rows.items() for name in ('original','scalar') for step in range(len(r['trace']))}
    seen={};timings={}
    for line in (source/'callbacks.jsonl').read_text().splitlines():
        r=json.loads(line);key=tuple(r['key']);identity=(key,r['implementation'],r['step'])
        if identity in seen or r['error'] is not None or not np.isfinite(r['wall_ms']) or r['wall_ms']<0:
            raise ValueError('Duplicate/failed/invalid callback')
        item=source_rows[key]['trace'][r['step']]
        if np.asarray(r['command']).tobytes()!=np.asarray(item['command']).tobytes():raise ValueError('Command bytes differ')
        seen[identity]=(r['command'],{k:v for k,v in r['info'].items() if k!='gate_ms'})
        for field,value in seen[identity][1].items():
            if item.get(field)!=value:raise ValueError('Original evidence differs: '+field)
        model=key[-1].split('_')[-1];timings.setdefault(model+'/'+r['implementation'],[]).append(r['wall_ms'])
    if set(seen)!=expected or len(source_rows)!=240:raise ValueError('Incomplete full callback population')
    for key,r in source_rows.items():
        for step in range(len(r['trace'])):
            if seen[(key,'original',step)]!=seen[(key,'scalar',step)]:raise ValueError('Paired full outputs differ')
    summary={};reductions={}
    for key,values in timings.items():
        a=np.array(values);summary[key]=dict(callbacks=len(a),sum_ms=float(a.sum()),p95_ms=float(np.quantile(a,.95)),max_ms=float(a.max()),over_500ms=int((a>500).sum()))
        for field,value in summary[key].items():
            if report['summary'][key][field]!=value:raise ValueError('Summary differs: '+key+'/'+field)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True,layout='constrained')
    for ax,seed in zip(axes,(4703,4704,4705)):
        for name,color,label in [('original','#7c8799','Original parameter gate'),('scalar','#137a72','Exact scalar implementation')]:
            a=np.sort(np.array(timings[str(seed)+'/'+name]));ax.plot(a,np.arange(1,len(a)+1)/len(a),color=color,label=label,lw=1.8)
        old=summary[str(seed)+'/original'];new=summary[str(seed)+'/scalar']
        reduction=100*(1-new['sum_ms']/old['sum_ms']);reductions[str(seed)]=reduction
        ax.axvline(500,color='#b24437',linestyle='--',lw=1)
        ax.set_xscale('log');ax.set_title(f'MAPPO seed {seed}\nTotal callback time reduced {reduction:.1f}%')
        ax.set_xlabel('Complete callback wall time (ms)');ax.grid(alpha=.18)
    axes[0].set_ylabel('Empirical cumulative fraction');axes[-1].legend(loc='lower right',fontsize=8)
    fig.suptitle('Exact replay of 240 existing SUMO trajectories',fontsize=12)
    fig.savefig(out/'latency.png',dpi=170);fig.savefig(out/'latency.svg');plt.close(fig)
    proof=dict(all_callback_population_and_original_fields_match=True,callbacks=len(seen),source_episodes=240,
        whole_controller_replays=480,recomputed_summary=summary,total_time_reduction_percent=reductions,
        extra_sumo_executions=0,extra_independent_initials=0,risk_certificate=False,
        independent_initials_in_reused_source=5,callback_data_sha256=sha(source/'callbacks.jsonl'),
        report_sha256=sha(source/'report.json'),analysis_source_sha256=sha(Path(__file__)),
        limitations='Correlated replay calls on shared desktop; no statistical independence, isolation, hard-real-time or efficacy claim.')
    with (out/'independent_audit.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':run()
