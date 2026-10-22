"""Describe the pilot's fixed conflict groups without selecting a new threshold."""
import argparse,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v14/development/v14_validation_average/episodes.json'


def analyze(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=json.loads(SOURCE.read_text())
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    rng=np.random.default_rng(14151);groups={}
    for group in ('low_conflict','high_conflict'):
        seeds=sorted({r['seed'] for r in rows if r['group']==group})
        ids=rng.integers(0,len(seeds),(10000,len(seeds)))
        result={'initials':len(seeds),'metrics':{}}
        for key in ('ego_cost','ego_time','ego_variation','other_time'):
            b=np.array([np.mean([lookup[s,m,'rollout'][key] for m in ('clean','uniform','bias','pulse')]) for s in seeds])
            l=np.array([np.mean([lookup[s,m,'policy_0'][key] for m in ('clean','uniform','bias','pulse')]) for s in seeds])
            result['metrics'][key]=dict(baseline=float(b.mean()),learned=float(l.mean()),
                gain_percent=float(100*(b.mean()-l.mean())/b.mean()),mean_improvement=float((b-l).mean()),
                ci95=np.quantile((b-l)[ids].mean(-1),[.025,.975]).tolist())
        groups[group]=result
    result=dict(source=str(SOURCE.relative_to(ROOT)),status='Descriptive development analysis; one training seed',
                group_rule='Frozen conflict_group from V12: minimum ego/neighbor ETA separation <= 1.25 s is high conflict',
                groups=groups,bootstrap=dict(seed=14151,draws=10000,unit='initial, averaging four modes'),
                caution='Group intervals are descriptive, not multiplicity-adjusted; no threshold was fitted from these results')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.7),layout='constrained')
    for ax,key,title in zip(axes,('ego_cost','ego_time','other_time'),('Ego cost','Ego arrival time','Other vehicles arrival time')):
        for i,group in enumerate(groups):
            r=groups[group]['metrics'][key];value=r['mean_improvement'];lo,hi=r['ci95']
            ax.errorbar(i,value,yerr=[[value-lo],[hi-value]],fmt='o',capsize=5,color=('#6b7280','#007f86')[i])
        ax.axhline(0,color='#999999',linewidth=.8,linestyle='--')
        ax.set_xticks([0,1],['Low conflict\n34 initials','High conflict\n94 initials'])
        ax.set_ylabel('Baseline minus learning (s)');ax.set_title(title);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Development only: one trained policy, paired initial-cluster 95% intervals\nPositive values favor learning; negative values indicate regression',fontsize=11)
    fig.savefig(out/'conflict_groups.png',dpi=180);fig.savefig(out/'conflict_groups.pdf');plt.close(fig)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/v14_conflict_development');a=p.parse_args();analyze(a.out)
