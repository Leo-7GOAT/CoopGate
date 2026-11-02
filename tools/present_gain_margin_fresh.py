"""Independent complete development aggregation and a fixed future candidate.

Selection uses development ONLY. No risk budget is reserved or certificate
issued;formal sampling/analysis still requires its own completed registration.
"""
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba
import numpy as np
import torch
from coopgate.mappo_deployment import load_deployment_actor

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def run():
    out=BASE/'gain_margin_fresh_analysis_20261102';out.mkdir(exist_ok=False)
    report_path=BASE/'gain_margin_fresh_development_20261102.json';report=json.loads(report_path.read_text())
    execution=BASE/'gain_margin_fresh_20261102_execution';receipt=json.loads((execution/'orchestrator_native_receipt.json').read_text())
    complete=json.loads((execution/'completion.json').read_text())
    if receipt['actual_native_exit_code']!=0 or not complete['all_actual_native_exits_zero'] or sha(report_path)!=complete['report_sha256']:
        raise ValueError('Observed terminal-zero complete family required')
    records=[]
    for name,identity in report['sources'].items():
        p=BASE/'development'/name
        if sha(p/'episodes.jsonl')!=identity['episodes_sha256'] or sha(p/'verification.json')!=identity['verification_sha256']:
            raise ValueError('Source data/verification changed')
        records.extend(json.loads(s) for s in (p/'episodes.jsonl').read_text().splitlines())
    lookup={tuple(r['key']):r for r in records};cases=sorted({tuple(r['key'][:4]) for r in records})
    methods=sorted({r['key'][-1] for r in records})
    if len(records)!=1040 or len(lookup)!=1040 or len(cases)!=80 or len({c[0] for c in cases})!=80 or len(methods)!=13:
        raise ValueError('Incomplete1040-outcome/80-initial family')
    independent={};adverse=[]
    for method in methods:
        adopted=harmful=failed=0;percent=[]
        for case in cases:
            row=lookup[case+(method,)]['row'];base=lookup[case+('baseline',)]['row']
            failure=any(not r['completed'] or r['aborted'] or r['collision'] or r['contract_violations'] for r in (row,base))
            failed+=failure;adopt=row['learned_action_steps']>0;adopted+=adopt
            dc=np.array(row['per_vehicle_cost'])-base['per_vehicle_cost'];dt=np.array(row['per_vehicle_finish'])-base['per_vehicle_finish']
            bad=bool(failure or np.any(dc>1e-9) or np.any(dt>1e-9));harmful+=bool(adopt and bad)
            percent.append(100*(base['team_cost']-row['team_cost'])/base['team_cost'])
            if adopt and bad:adverse.append(dict(case=list(case),method=method,cost_difference=dc.tolist(),time_difference=dt.tolist()))
        mean=math.fsum(percent)/80.;old=report['methods'][method]
        if old['adopted']!=adopted or old['harmful_adopted']!=harmful or abs(mean-old['mean_paired_percentage_cost_gain'])>1e-12:
            raise ValueError('Independent adoption/harm/paired-gain aggregation differs')
        independent[method]=dict(independent_initials=80,adopted=adopted,harmful_adopted=harmful,
            failed_or_failed_baseline=failed,mean_paired_percentage_cost_gain=mean)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    xs=[0,.25,.5,1.];floors=(0,25,50,100)
    for seed,color,marker in ((4703,'#137a72','o'),(4704,'#496ea4','s'),(4705,'#b6652e','^')):
        values=[independent[f'gain_{seed}_{f}'] for f in floors]
        series=([100*v['adopted']/80 for v in values],
            [v['mean_paired_percentage_cost_gain'] for v in values],
            [100*v['harmful_adopted']/v['adopted'] if v['adopted'] else np.nan for v in values])
        for ax,y in zip(axes,series):ax.plot(xs,y,color=color,marker=marker,lw=1.6,label=f'MAPPO {seed}')
    for ax,label in zip(axes,('Episodes adopting learning (%)','Mean paired team-cost gain (%)','Observed harmful / adopted (%)')):
        ax.set_ylabel(label);ax.set_xlabel('Predicted gain floor (%)');ax.set_xticks(xs);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[2].set_ylim(-.5,12.5)
    fig.suptitle('Fresh development on 80 distinct initials   No risk certification',fontsize=12)
    fig.savefig(out/'tradeoff.png',dpi=180);fig.savefig(out/'tradeoff.svg');plt.close(fig)
    proof=dict(all_complete_development_counts_independently_match=True,actual_sumo_executions=1040,independent_initials=80,
        per_method=independent,all_adverse_outcomes=adverse,report_sha256=sha(report_path),source_sha256=sha(Path(__file__)),
        risk_certificate=False,inference='Descriptive balanced factorial development;no formal confidence bounds or alpha allocation.')
    write(out/'independent_audit.json',proof)
    loaded={str(s):load_deployment_actor(BASE/f'development/joint_{s}/policy.pt') for s in (4703,4704,4705)}
    for seed in loaded:
        item=independent[f'gain_{seed}_25']
        if item['failed_or_failed_baseline'] or item['harmful_adopted'] or item['adopted']/80<.20 or item['mean_paired_percentage_cost_gain']<=.5:
            raise ValueError('Selected developmental candidate does not warrant the registered formal follow-up')
    candidate=dict(controller='PrunedGainMarginGate',common_gain_floor=.0025,actor_seeds=[4703,4704,4705],
        actor_identities={s:value[1] for s,value in loaded.items()},
        algorithm='genuine joint MAPPO with centralized critic and independent local actors;PPO training independently audited',
        initialization='Disclosed legacy-PI warm start via new single-PPO4601,then joint PPO;not from-scratch model-free learning',
        selection='Lowest registered positive floor,chosen using development before ANY new formal data. Fixes known fixture;all other floors/negative outcomes retained.',
        development_evidence_sha256=sha(out/'independent_audit.json'),
        controller_source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'src/coopgate').glob('*.py'))},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,numba=numba.__version__),
        formal_calibration_initials=3072,formal_confirmation_initials=1024,
        required_mixture_conditional_harm_upper=.05,required_strong_lag_conditional_harm_upper=.05,
        required_adoption_lower=.20,required_mean_paired_percentage_gain_t_lower_strictly_above=.5,
        required_observed_baseline_or_candidate_failures=0,same_adoption_controls_required=True,
        error_allocation_attempt_reserved=None,formal_experiment_started=False,risk_certificate=False,paper_ready=False,
        unresolved=['Audit predecessor error-allocation ledger before reserving new attempt',
            'Formal fixed-N collector retains and independently verifies adverse outcomes;never resamples failures',
            'Actual matching-rate controls with independent controller coins and unchanged baseline',
            'Worst callbacks remain above500ms in shared four-worker fresh development;no hard-real-time claim',
            'Generic raw MAPPO and consistent MAPPO-over-IPPO superiority unproved;single4603 interval failure retained',
            'Old13-page manuscript is PI;new MAPPO manuscript must not inherit its certificate/results'])
    write(BASE/'gain_margin_fixed_candidate_20261102.json',candidate)
    print(json.dumps(dict(independent_audit_passed=True,fixed_common_floor=.0025,formal_started=False,risk_certificate=False,
        per_method={k:v for k,v in independent.items() if k.endswith('_25')},adverse_outcome_records=len(adverse))),flush=True)


if __name__=='__main__':run()
