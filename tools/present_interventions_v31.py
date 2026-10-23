"""Present the entire completed intervention and trajectory-identity scope."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from execute_prospective_v23 import exclusive_json, utc
from profile_phase_runtime_v24 import digest

ROOT = Path(__file__).resolve().parents[1]
INTERVENTIONS = ROOT/'artifacts/v31/actuator_interventions_roundoff'
IDENTITY = ROOT/'artifacts/v31/performance_identity'
OUT = ROOT/'artifacts/v31/intervention_presentation'
PLANTS = ('nominal','gain_drag','lag_mild','lag_strong')
LABELS = ('Nominal','Gain / drag','Mild lag','Strong lag')


def report(folder):
    path = folder/'analysis/report.json'
    receipt = json.loads((folder/'analysis/sha256.json').read_text())
    if digest(path) != receipt['report']:
        raise ValueError('Completed report checksum differs')
    return json.loads(path.read_text())


def main():
    interventions,identity = report(INTERVENTIONS),report(IDENTITY)
    exit = json.loads((INTERVENTIONS/'execution/observation.json').read_text())
    if (exit['exit_code'] != 0 or interventions['actual_sumo_runs'] != 698
            or interventions['admissions'] != 349 or identity['source_episodes'] != 1920):
        raise ValueError('Both complete scopes and actual successful execution required')
    counts,changed = [],[]
    for plant in PLANTS:
        group = interventions['by_plant'][plant]
        counts.append([group['complete_pairs']-group['local_regressions'],
                       group['local_regressions'],group['admissions']-group['complete_pairs']])
        cases = [c for c in interventions['cases'] if c['case'][4] == plant]
        changed.append([sum(c['nominal_model_pair']['admissible_performance_evidence'] and
                            not c['actual_model_pair']['admissible_performance_evidence'] for c in cases),
                        sum(not c['nominal_model_pair']['admissible_performance_evidence'] and
                            c['actual_model_pair']['admissible_performance_evidence'] for c in cases)])
    accounting = []
    for plant in PLANTS:
        group = identity['by_plant'][plant]
        accounting.append([group['matched'],group['assessable']-group['matched'],group['source_episodes']-group['assessable']])
    counts,changed,accounting = map(np.asarray,(counts,changed,accounting))
    OUT.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes = plt.subplots(1,3,figsize=(16,5.8))
    x = np.arange(4)
    categories = ('Complete: no local regression','Complete: local regression','Incomplete pair')
    colors = ('#269c8b','#d74c4c','#b7bdc5')
    bottom = np.zeros(4)
    for i,(label,color) in enumerate(zip(categories,colors)):
        axes[0].bar(x,counts[:,i],bottom=bottom,color=color,label=label,width=.66)
        for j,value in enumerate(counts[:,i]):
            if value:
                axes[0].text(j,bottom[j]+value/2,str(value),ha='center',va='center',fontsize=9)
        bottom += counts[:,i]
    axes[0].set_title('A  Actual paired continuations\nAll 349 archived learning admissions',loc='left',fontweight='bold',pad=13)
    axes[0].set_ylabel('Admissions (reused, dependent conditions)')
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.18),frameon=False,fontsize=9)
    labels = ('Nominal model passes; actual model rejects','Nominal model rejects; actual model passes')
    for i,color in enumerate(('#d74c4c','#377bb5')):
        offset = (i-.5)*.32
        axes[1].bar(x+offset,changed[:,i],width=.3,color=color,label=labels[i])
        for j,value in enumerate(changed[:,i]):
            axes[1].text(j+offset,value+.12,str(value),ha='center',va='bottom',fontsize=9)
    axes[1].set_ylim(0,max(1,changed.max())*1.22+.3)
    axes[1].set_title('B  Model-dependent performance judgment\nSame state and realized external path',loc='left',fontweight='bold',pad=13)
    axes[1].set_ylabel('Changed local classifications')
    axes[1].legend(loc='upper left',bbox_to_anchor=(0,-.18),frameon=False,fontsize=9)
    bottom = np.zeros(4)
    for i,(label,color) in enumerate(zip(('Identity verified','Identity mismatch','Incomplete: no identity claim'),colors)):
        axes[2].bar(x,accounting[:,i],bottom=bottom,color=color,label=label,width=.66)
        for j,value in enumerate(accounting[:,i]):
            if value:
                if value < 30:
                    axes[2].text(j,bottom[j]+value/2,str(value),ha='center',va='center',fontsize=8,
                        bbox=dict(facecolor='white',edgecolor='none',pad=.3))
                else:
                    axes[2].text(j,bottom[j]+value/2,str(value),ha='center',va='center',fontsize=9)
        bottom += accounting[:,i]
    axes[2].set_title('C  Local-to-trajectory accounting\nAll 1,920 original source episodes',loc='left',fontweight='bold',pad=13)
    axes[2].set_ylabel('Source episodes (same 160 initial conditions)')
    axes[2].legend(loc='upper left',bbox_to_anchor=(0,-.18),frameon=False,fontsize=9)
    for ax in axes:
        ax.set_xticks(x,LABELS,rotation=15)
        ax.set_axisbelow(True)
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('Actuator-aware counterfactual verification and performance accounting',x=.055,ha='left',fontsize=16,fontweight='bold',y=.97)
    fig.text(.055,.025,'Offline diagnostic with known parameters and realized future innovation. 698 recorded SUMO runs; 404 earlier executions retained.\nAll original metrics and failures preserved. No new independent initials, deployed controller, or risk certificate.',fontsize=10,color='#404852')
    fig.subplots_adjust(left=.055,right=.985,top=.77,bottom=.31,wspace=.3)
    for ext in ('png','svg'):
        fig.savefig(OUT/('intervention_mechanisms.'+ext),dpi=180)
    plt.close(fig)
    exclusive_json(OUT/'data.json',dict(plants=PLANTS,local_pair_categories=categories,local_counts=counts.tolist(),
        changed_classification_labels=labels,changed_classifications=changed.tolist(),trajectory_identity_counts=accounting.tolist()))
    exclusive_json(OUT/'provenance.json',dict(utc=utc(),inputs={str(folder.relative_to(ROOT)):digest(folder/'analysis/report.json') for folder in (INTERVENTIONS,IDENTITY)},
        source_sha256=digest(Path(__file__)),files={p.name:digest(p) for p in OUT.iterdir() if p.is_file()},all_cells_and_failures_included=True))
    print(json.dumps(dict(output=str(OUT),source_episodes=1920,admissions=349)),flush=True)


if __name__ == '__main__':
    main()
