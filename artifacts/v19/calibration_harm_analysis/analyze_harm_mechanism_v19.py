"""Describe all completed harm replays without reading the final test."""
from collections import Counter
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from coopgate.advantage_gate_v18 import baseline_suffix
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import verify_manifest
from study_gate_v19 import read_protocol

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v19/calibration_harm_replay'
OUTPUT=ROOT/'artifacts/v19/calibration_harm_analysis'


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve earlier mechanism analysis')
    verify_manifest(SOURCE)
    read_protocol(ROOT/'artifacts/v19/risk_validation')
    audit=json.loads((SOURCE/'audit.json').read_text())
    if not audit['all_selected_non_timing_metrics_match'] or not audit['all_hindsight_identities_match']:
        raise ValueError('Complete replay and potential identity verification required')
    packed=parameters(MergeConfig())
    totals=Counter()
    per_model={name:Counter() for name in sorted(audit['per_model_harmful_runs'])}
    events=[]
    for case in audit['cases']:
        if case['controller']=='baseline':
            continue
        number=case['case_index']
        row=json.loads((SOURCE/f'row_{number}.json').read_text())
        trace=json.loads((SOURCE/f'trace_{number}.json').read_text())
        mechanism=json.loads((SOURCE/f'mechanism_{number}.json').read_text())
        per_model[case['controller']]['harmful_runs']+=1
        for step in mechanism['steps']:
            if not step['admitted']:
                continue
            k=step['step']
            state=np.asarray(row['initial'] if k==0 else trace[k-1]['x'])
            previous=np.zeros(3) if k==0 else np.asarray(trace[k-1]['command'])
            command=np.asarray(trace[k]['command'])
            _,base,_,_=baseline_suffix(state,previous,k,packed)
            changed=np.abs(command-base)>1e-10
            actual=np.asarray(step['hindsight_bellman_difference'])
            bank=np.asarray(step['bank_componentwise_max'])
            if np.any(bank>1e-10):
                raise ValueError('Archived bank admitted a positive component')
            positive=actual>1e-9
            bad=bool(positive.any())
            indirect_only=bool(bad and not np.any(actual[changed]>1e-9))
            counts=dict(admitted_steps=1,positive_admission_steps=int(bad),positive_components=int(positive.sum()),
                        positive_components_with_zero_bank_slack=int((positive&(bank>=-1e-10)).sum()),
                        positive_components_on_unchanged_vehicle_command=int((positive&~changed[:,None]).sum()),
                        harmful_steps_only_on_unchanged_vehicle_commands=int(indirect_only))
            totals.update(counts)
            per_model[case['controller']].update(counts)
            events.append(dict(case_index=number,initial_index=case['initial_index'],seed=row['seed'],
                               scenario=row['scenario'],disturbance=case['disturbance'],controller=case['controller'],
                               step=k,composition=step['composition'],state_before=state.tolist(),
                               previous_command=previous.tolist(),chosen_command=command.tolist(),baseline_command=base.tolist(),
                               changed_vehicle_commands=changed.tolist(),bank_componentwise_max=bank.tolist(),
                               hindsight_difference=actual.tolist(),positive_hindsight_residual=bad,
                               harm_only_on_currently_unchanged_vehicle_commands=indirect_only))
    if totals['positive_admission_steps']!=audit['positive_hindsight_admission_steps'] or len(events)!=totals['admitted_steps']:
        raise ValueError('Hindsight event accounting differs')
    bad=[event for event in events if event['positive_hindsight_residual']]
    report=dict(purpose='Post hoc mechanism analysis of all harmful calibration replays',
                source_sumo_replays=audit['actual_sumo_replays'],harmful_runs=audit['source_harmful_runs'],
                totals=dict(totals),per_model={key:dict(value) for key,value in per_model.items()},
                bad_step_range=[min(e['step'] for e in bad),max(e['step'] for e in bad)],
                all_bad_steps_have_a_vehicle_before_or_in_conflict_zone=all(np.any(np.asarray(e['state_before'])[:,0]<=6.8) for e in bad),
                bad_compositions=dict(Counter(str(e['composition']) for e in bad)),
                added_independent_risk_samples=0,final_confirmation_outcomes_read=False,policy_changes=0,
                selection_scope='All 46 harmful calibration runs and their paired baselines. Frequencies here are conditional on this selected failure set, not population risk estimates.',
                source_audit_sha256=digest(SOURCE/'audit.json'),source_manifest_sha256=digest(SOURCE/'sha256.json'),
                analyzer_sha256=digest(__file__))
    OUTPUT.mkdir()
    save(OUTPUT/'analysis.json',report)
    save(OUTPUT/'admission_events.json',events)
    shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.0),gridspec_kw={'width_ratios':[1.15,1]})
    for metric,label,color,marker in [(0,'Cost component','#c85b19','o'),(1,'Time component','#2569a5','^')]:
        x=np.asarray([np.asarray(e['bank_componentwise_max'])[:,metric] for e in events]).ravel()
        y=np.asarray([np.asarray(e['hindsight_difference'])[:,metric] for e in events]).ravel()
        axes[0].scatter(x,y,s=23,alpha=.65,color=color,marker=marker,label=label,edgecolors='none')
    axes[0].axhline(0,color='#777777',lw=.8,ls='--')
    axes[0].axvline(0,color='#777777',lw=.8,ls='--')
    axes[0].set(xlabel='Maximum predicted difference across 17 tapes',ylabel='Hindsight baseline-value difference',
                title=f"(a) {len(events)} admitted steps; all six components")
    axes[0].legend(frameon=False,fontsize=9,loc='lower left')
    models=list(per_model)
    indirect=[per_model[m]['harmful_steps_only_on_unchanged_vehicle_commands'] for m in models]
    direct=[per_model[m]['positive_admission_steps']-indirect[i] for i,m in enumerate(models)]
    axes[1].bar(range(3),direct,color='#43859f',label='Loss includes a changed-command vehicle')
    axes[1].bar(range(3),indirect,bottom=direct,color='#a76f9c',label='Loss only on unchanged-command vehicles')
    for i,m in enumerate(models):
        axes[1].text(i,direct[i]+indirect[i]+.35,str(direct[i]+indirect[i]),ha='center')
    axes[1].set(xticks=range(3),xticklabels=[m.rsplit('_',1)[1] for m in models],xlabel='Fixed training model',
                ylabel='Harmful admission steps',title='(b) Coupled effects at the admission step',ylim=(0,max(direct[i]+indirect[i] for i in range(3))*1.38))
    axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    fig.text(.5,.02,'Post hoc diagnosis of 46 harmful calibration runs; not an independent risk or efficacy estimate.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1))
    fig.savefig(OUTPUT/'hindsight_admission_diagnosis.png')
    fig.savefig(OUTPUT/'hindsight_admission_diagnosis.pdf')
    plt.close(fig)
    save(OUTPUT/'sha256.json',{p.relative_to(OUTPUT).as_posix():digest(p) for p in OUTPUT.rglob('*') if p.is_file() and p.name!='sha256.json'})
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    run()
