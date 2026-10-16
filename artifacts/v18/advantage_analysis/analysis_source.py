"""Paired initial-cluster analysis of complete nominal gating studies."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def analyze(study,out):
    study,out=Path(study),Path(out)
    p=json.loads((study/'protocol.json').read_text());rows=json.loads((study/'episodes.json').read_text())
    methods=p['methods'];modes=p['modes'];n=p['n_initials'];seeds=range(p['seed_base'],p['seed_base']+n)
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    assert len(rows)==len(lookup)==p['expected_sumo_runs']
    assert set(lookup)=={(s,m,c) for s in seeds for m in modes for c in methods}
    arrays={name:np.array([[np.column_stack((lookup[s,m,name]['per_vehicle_cost'],lookup[s,m,name]['per_vehicle_finish']))
        for m in modes] for s in seeds]) for name in methods}
    groups=np.array([lookup[s,modes[0],'baseline']['group'] for s in seeds])
    results={}
    for name in methods:
        data=[r for r in rows if r['controller']==name];difference=arrays['baseline']-arrays[name]
        selected=np.ones(n,bool)
        if name=='baseline':selected[:]=False
        elif 'component_learning_mask' in data[0]:
            selected=np.array([any(lookup[s,m,name]['component_learning_mask']) for s in seeds for m in modes]).reshape(n,len(modes)).any(1)
        elif name=='nominal_action_gate':
            selected=np.array([lookup[s,m,name]['learned_action_steps']>0 for s in seeds for m in modes]).reshape(n,len(modes)).any(1)
        harmed=np.any(difference < -1e-9,axis=(1,2,3))
        summaries={}
        for group,mask in [('all',np.ones(n,bool))]+[(g,groups==g) for g in sorted(set(groups))]:
            count=int(mask.sum());d=difference[mask];delta=d.mean(1);team=delta[:,:,0].mean(1)
            rng=np.random.default_rng(18190001);indices=rng.integers(0,count,(5000,count))
            summaries[group]=dict(initials=count,baseline_cost=float(arrays['baseline'][mask,:,:,0].mean()),
                method_cost=float(arrays[name][mask,:,:,0].mean()),mean_team_gain=float(team.mean()),
                team_gain_percent=100*float(team.mean()/arrays['baseline'][mask,:,:,0].mean()),
                paired_team_gain_ci95=np.quantile(team[indices].mean(1),[.025,.975]).tolist(),
                mean_individual_gain=delta.mean(0).tolist(),
                paired_individual_gain_ci95=np.quantile(delta[indices].mean(1),[.025,.975],axis=0).tolist(),
                admitted_initials=int(selected[mask].sum()),harmful_initials=int(harmed[mask].sum()),
                conditional_empirical_harm=float(harmed[mask&selected].mean()) if (mask&selected).any() else None,
                worst_cost_loss_quantiles=np.quantile(np.maximum((-d[:,:,:,0]).max((1,2)),0),[.5,.9,.95,1.]).tolist())
        results[name]=dict(groups=summaries,completed=sum(r['completed'] for r in data),runs=len(data),
            collisions=sum(r['collision'] for r in data),aborted=sum(r['aborted'] for r in data),
            contract_violations=sum(r['contract_violations'] for r in data),
            p95_computation_ms=float(np.percentile([t for r in data for t in r['controller_ms']],95)))
    mechanisms={}
    if (study/'decisions.json').exists():
        decisions=json.loads((study/'decisions.json').read_text())
        for name in ('independent_nominal','joint_nominal'):
            masks=np.array([d['selected_masks'][name] for d in decisions])
            clean=arrays[name][:,modes.index('clean')]-arrays['baseline'][:,modes.index('clean')]
            mechanisms[name]=dict(two_or_more_selected=int((masks.sum(1)>=2).sum()),
                clean_any_vehicle_harmed_initials=int(np.any(clean>1e-9,axis=(1,2)).sum()),
                clean_selected_vehicle_harmed_initials=int(np.any((clean>1e-9)&masks[:,:,None],axis=(1,2)).sum()),
                clean_selected_vehicle_harm_seeds=[s for s,h in zip(seeds,np.any((clean>1e-9)&masks[:,:,None],axis=(1,2))) if h])
    out.mkdir(parents=True,exist_ok=False)
    record=dict(purpose='Development estimates conditional on a fixed trained model; no formal acceptance or calibrated risk claim',
        sampling_unit='Initial state, with four modes in one cluster',n_initials=n,results=results,composition_mechanisms=mechanisms,
        source_sha256={name:hashlib.sha256((study/name).read_bytes()).hexdigest() for name in ('protocol.json','episodes.json')})
    (out/'analysis.json').write_text(json.dumps(record,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[k for k in methods if k!='baseline' and not k.startswith('isolated_')]
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    for i,name in enumerate(selected):
        data=results[name]['groups']['all'];g=np.array(data['mean_individual_gain'])[:,0]
        values=np.r_[g,data['mean_team_gain']]
        bounds=np.array(data['paired_individual_gain_ci95'])[:,:,0]
        bounds=np.column_stack((bounds,np.array(data['paired_team_gain_ci95'])))
        axes[0].errorbar(np.arange(4)+(i-(len(selected)-1)/2)*.16,values,
            yerr=np.maximum(0,np.vstack((values-bounds[0],bounds[1]-values))),fmt='o',capsize=3,label=name)
        axes[1].bar(i-.17,data['admitted_initials']/n,width=.32,color='#3b82a0',label='Any adoption' if i==0 else None)
        axes[1].bar(i+.17,data['harmful_initials']/n,width=.32,color='#d07245',label='Any cost/time regression' if i==0 else None)
    axes[0].set_xticks(range(4),['Main 0','Ramp 1','Main 2','Team']);axes[0].axhline(0,color='#777',linestyle='--',linewidth=.8)
    axes[0].set_ylabel('Baseline minus method cost (s)');axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    axes[1].set_xticks(range(len(selected)),[k.replace('_','\n') for k in selected],fontsize=8)
    axes[1].set_ylim(0,1.05);axes[1].set_ylabel('Fraction of initial-state clusters');axes[1].legend(fontsize=8)
    fig.suptitle(f'Development: {n} paired initials, four modes; nominal gates\n95% paired bootstrap intervals; performance regression is not a collision',fontsize=11)
    fig.savefig(out/'gates.png',dpi=180);fig.savefig(out/'gates.pdf');plt.close(fig)
    print(json.dumps({k:v['groups']['all'] for k,v in results.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('--out',required=True)
    a=p.parse_args();analyze(a.study,a.out)
