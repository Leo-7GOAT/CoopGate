# original source-bound plotting sha256=fe05684f12b6f4906f64fcdaa0fee6b490c97104322878d70991246645c36ffd
"""Render finalized formal evidence,without new tests,selection or inference.

No outcome access precedes exact complete native receipts. Every plotted
number is copied from the original unchanged final certificate/comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'artifacts/mappo_ctde_20261101/formal_mappo_attempt5_powered_20261103'
SEEDS=('4703','4704','4705')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


from mappo_completed_powered_attempt5 import complete_evidence


def render(out):
    if out.exists() or not out.resolve().is_relative_to(ROOT/'artifacts/mappo_ctde_20261101'):
        raise ValueError('Exclusive new output in authorized MAPPO artifact scope required')
    cert,comp=complete_evidence();out.mkdir(parents=True,exist_ok=False)
    if cert['fixed_models']!=[4703,4704,4705] or cert['common_gain_floor']!=.0025:
        raise ValueError('Wrong frozen target family')
    plt.rcParams.update({'font.size':14,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    colors=('#17659b','#d88824','#4b8b50');files=[]
    def save(fig,name):
        for suffix in ('.png','.svg'):
            path=out/(name+suffix);fig.savefig(path,bbox_inches='tight')
            files.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size))
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,4.4));x=np.arange(3);risk=cert['primary_risk']['members']
    for offset,kind,label,color in ((-.16,'mixture','Mixture','#17659b'),(.16,'strong_lag','Strong lag','#d88824')):
        values=[100*risk[s+':'+kind]['pointwise_risk_upper'] for s in SEEDS]
        axes[0].bar(x+offset,values,.30,label=label,color=color)
    axes[0].axhline(5,color='#b62828',linestyle='--',label='Limit 5%');axes[0].set_ylabel('Harm upper bound (%)')
    axes[0].set_title('Conditional risk')
    adoption=[100*cert['adoption']['members'][s]['adoption_lower'] for s in SEEDS]
    axes[1].bar(x,adoption,color=colors);axes[1].axhline(20,color='#b62828',linestyle='--')
    axes[1].set_ylabel('Adoption lower bound (%)');axes[1].set_title('IID adoption')
    gain=[cert['paired_gain']['members'][s]['gain_lower_percent'] for s in SEEDS]
    axes[2].bar(x,[np.nan if v is None else v for v in gain],color=colors)
    axes[2].axhline(.5,color='#b62828',linestyle='--');axes[2].set_ylabel('Gain lower bound (%)');axes[2].set_title('IID utility')
    for ax in axes:ax.set_xticks(x,SEEDS);ax.set_xlabel('MAPPO seed');ax.grid(axis='y',alpha=.18)
    verdict='PASS' if cert['joint_deployment_certified'] else 'NOT CERTIFIED'
    fig.suptitle('Fixed three-model joint decision: '+verdict,fontsize=16,y=1.05)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False,fontsize=12.5)
    fig.text(.5,-.07,'Original component bounds; alpha=0.05/30; joint acceptance by IUT.\nGain t bound: normal IID exact; otherwise asymptotic.',ha='center',fontsize=12.5)
    fig.tight_layout(rect=(0,0,1,.86));save(fig,'formal_acceptance')
    summary=comp['confirmation_methods'];prefixes=('','nominal_','thinned_','mps_','mamps_')
    labels=('Gate','Nominal','Thinned','MPS','MAMPS')
    all_gains=[summary[p+s]['mean_paired_percentage_gain'] for s in SEEDS for p in prefixes]
    fig,axes=plt.subplots(3,3,figsize=(10,10),sharex=True)
    for column,seed in enumerate(SEEDS):
        rows=[summary[p+seed] for p in prefixes];xs=np.arange(len(rows))
        shown=list(labels)
        if comp['calibration_matching']['probabilities'][seed] is None:shown[2]='INACTIVE'
        adopted=[r['adopted']/1024*100 for r in rows]
        harm=[np.nan if not r['adopted'] else r['harmful_adopted']/r['adopted']*100 for r in rows]
        axes[0,column].bar(xs,adopted,color=colors[column]);axes[0,column].set_title('MAPPO '+seed)
        axes[1,column].bar(xs,harm,color=colors[column]);axes[1,column].axhline(5,color='#b62828',linestyle='--')
        axes[2,column].bar(xs,[r['mean_paired_percentage_gain'] for r in rows],color=colors[column]);axes[2,column].axhline(0,color='gray',linewidth=.7)
        for row in range(3):axes[row,column].grid(axis='y',alpha=.18)
        axes[0,column].set_ylim(0,105);axes[1,column].set_ylim(0,110)
        axes[1,column].set_yticks((0,25,50,75,100))
        axes[2,column].set_ylim(min(0,min(all_gains))-.25,max(0,max(all_gains))+.5)
        axes[2,column].set_xticks(xs,shown,rotation=40,ha='right')
        for index,r in enumerate(rows):
            axes[1,column].annotate(str(r['harmful_adopted'])+'/'+str(r['adopted']),(index,0 if not r['adopted'] else 100*r['harmful_adopted']/r['adopted']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=12.5,rotation=45)
    axes[0,0].set_ylabel('Adoption (%)');axes[1,0].set_ylabel('Harm / adopted (%)');axes[2,0].set_ylabel('Paired gain (%)')
    fig.suptitle('All 1,024 IID confirmation initials, including failures',fontsize=16)
    fig.text(.5,-.01,'Descriptive points; no new certificate. Harm protects every vehicle.\nThinning uses calibration counts; actual adoption rates need not match.',ha='center',fontsize=12.5)
    fig.tight_layout(rect=(0,.04,1,.97));save(fig,'formal_control_comparisons')
    fig,axes=plt.subplots(1,3,figsize=(10,4.8))
    lower_time=min(summary[p+s]['callback_p95_ms'] for s in SEEDS for p in prefixes)*.7
    upper_time=max(summary[p+s]['callback_max_ms'] for s in SEEDS for p in prefixes)*1.4
    for column,seed in enumerate(SEEDS):
        rows=[summary[p+seed] for p in prefixes];xs=np.arange(len(rows))
        shown=list(labels)
        if comp['calibration_matching']['probabilities'][seed] is None:shown[2]='INACTIVE'
        axes[column].bar(xs-.17,[r['callback_p95_ms'] for r in rows],.32,label='P95',color=colors[column])
        axes[column].bar(xs+.17,[r['callback_max_ms'] for r in rows],.32,label='Maximum',color='#9b9b9b')
        axes[column].axhline(500,color='#b62828',linestyle='--');axes[column].set_yscale('log')
        axes[column].set_ylim(lower_time,upper_time)
        axes[column].set_xticks(xs,shown,rotation=40,ha='right');axes[column].set_title('MAPPO '+seed);axes[column].grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Callback time (ms, log scale)');axes[0].legend(fontsize=12.5)
    fig.suptitle('Complete online callbacks on the shared desktop',fontsize=16)
    fig.text(.5,-.08,'Every method executes and times its own full callback.\nNo communication latency, deadline cutoff or hard-real-time claim.',ha='center',fontsize=12.5)
    fig.tight_layout();save(fig,'formal_callback_cost')
    report=dict(plotter_source_sha256=sha(Path(__file__)),source_certificate_sha256=sha(STUDY/'analysis/certificate.json'),source_comparisons_sha256=sha(STUDY/'analysis/comparisons.json'),
        owner_receipt_sha256=sha(STUDY/'orchestrator_native_receipt.json'),joint_deployment_certified=cert['joint_deployment_certified'],
        matching_probabilities_valid=comp['calibration_matching']['all_matching_valid'],
        actual_sumo_runs=45056,independent_initials=5120,new_inference=False,selection_on_formal_outcomes=False,
        partial_outcomes_read=False,figures=files,visual_review_required=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();render(a.out)
