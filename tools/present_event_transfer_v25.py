"""Scientific plot of every phase05 event-change cell; no risk certification."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from profile_phase_runtime_v24 import digest,save

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v25/matched_event_transfer'
OUT=ROOT/'artifacts/v25/matched_event_transfer_presentation'


def main():
    manifest=json.loads((SOURCE/'sha256.json').read_text())
    for name,expected in manifest.items():
        if digest(SOURCE/name)!=expected:raise ValueError('Changed event audit')
    data=json.loads((SOURCE/'report.json').read_text())
    rows=[r for r in data['cells'] if r['family']=='phase05']
    scenes=['nominal','dense','slow_yield','asymmetric_speed','tight_headway']
    plants=['gain_drag','lag_mild','lag_strong'];models=['1802','1812','1822']
    lookup={(r['scenario'],r['target_plant'],r['model']):r for r in rows}
    if len(lookup)!=45 or len(rows)!=45:raise ValueError('All fixed phase05 cells required')
    fig,axes=plt.subplots(1,3,figsize=(10.2,4.6),layout='constrained',sharey=True)
    for ax,model in zip(axes,models):
        cells=[[lookup[scene,plant,model] for plant in plants] for scene in scenes]
        values=np.array([[r['coupled_event_mismatch'] for r in line] for line in cells])*100
        im=ax.imshow(values,cmap='YlOrRd',vmin=0,vmax=50,aspect='auto')
        for i,line in enumerate(cells):
            for j,r in enumerate(line):
                matrix=np.asarray(r['transitions'])
                k=int(matrix.sum()-matrix.trace())
                label=f'{k}/32'+('*' if r['target_error_exceeded'] else '')
                ax.text(j,i,label,ha='center',va='center',fontsize=10,color='white' if values[i,j]>32 else '#222222')
        ax.set_xticks(range(3),['Gain/drag','Mild lag','Strong lag'],rotation=25,ha='right',fontsize=9)
        ax.set_yticks(range(5),['Ordinary','Dense','Slow yield','Unequal speed','Tight gap'])
        ax.set_title('Fixed model '+model,fontsize=11)
    fig.colorbar(im,ax=axes,label='Paired event changes (%)',shrink=.8)
    fig.suptitle('Phase05: changes in adoption/harm category vs nominal plant\n'
        'Same 32 initials per scene; all models shown; * physical error bound exceeded\n'
        'Descriptive development evidence, not population shift bounds or certification',fontsize=10)
    OUT.mkdir(exist_ok=False)
    fig.savefig(OUT/'event_changes.png',dpi=180)
    fig.savefig(OUT/'event_changes.svg')
    plt.close(fig)
    totals=[]
    for plant in plants:
        cs=[r for r in rows if r['target_plant']==plant]
        totals.append(dict(plant=plant,cells=len(cs),cells_with_changed_events=sum(r['coupled_event_mismatch']>0 for r in cs),
            controller_runs=sum(r['paired_initials'] for r in cs),
            target_failures=sum(r['target_failures'] for r in cs),
            target_error_exceeded=sum(r['target_error_exceeded'] for r in cs),
            counts_are_repeated_across_three_fixed_models=True))
    save(OUT/'summary.json',dict(source_sha256=digest(SOURCE/'report.json'),cells=45,by_plant=totals,
        risk_certified=False,new_independent_initials=0,all_fixed_models_shown=True))
    save(OUT/'sha256.json',{p.name:digest(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(dict(cells=45,by_plant=totals)),flush=True)


if __name__=='__main__':main()
