"""Figures from completed, sealed timing and online-integration studies only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUFFER = ROOT/'artifacts/v29/buffered_scope_recovery_20261023'
ONLINE = ROOT/'artifacts/v32/online_actuator_preflight'
OUT = ROOT/'artifacts/v32/completed_evidence_presentation'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    buffered,online = read(BUFFER/'analysis/report.json'),read(ONLINE/'analysis/report.json')
    assert buffered['actual_sumo_runs']==6912 and online['actual_sumo_runs']==120
    assert read(BUFFER/'execution/exit.json')['exit_code']==0
    assert read(ONLINE/'completion_verification.json')['report_sha256']==sha(ONLINE/'analysis/report.json')
    assert read(BUFFER/'recovery_accounting.json')['source_report_sha256']==sha(BUFFER/'analysis/report.json')
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','figure.dpi':150})
    colors = ['#737f8c','#157a9b']
    groups = []
    for label,cells in [('All',buffered['cells']),('Development',[c for c in buffered['cells'] if c['study']=='phase']),
                        ('Transfer',[c for c in buffered['cells'] if c['study']=='transfer'])]:
        row = dict(group=label,paired_episodes=sum(c['episodes'] for c in cells),aborts=sum(c['aborted'] for c in cells))
        for kind in ('reference','buffered'):
            calls = sum(c[kind]['calls'] for c in cells)
            total = sum(c[kind]['sum_ms'] for c in cells)
            late = sum(c[kind]['over_500ms'] for c in cells)
            row[kind] = dict(calls=calls,sum_ms=total,mean_ms=total/calls,over_500ms=late,
                over_500ms_percent=100*late/calls,max_ms=max(c[kind]['p50_p95_p99_max_ms'][3] for c in cells))
        row['total_time_reduction_percent']=100*(1-row['buffered']['sum_ms']/row['reference']['sum_ms'])
        groups.append(row)
    fig,axes = plt.subplots(1,2,figsize=(11,4.7))
    x=np.arange(3);width=.35
    for ki,kind in enumerate(('reference','buffered')):
        for ax,field in zip(axes,('mean_ms','over_500ms_percent')):
            values=[r[kind][field] for r in groups]
            bars=ax.bar(x+(ki-.5)*width,values,width,color=colors[ki],label=kind.title())
            ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=9)
            ax.set_xticks(x,[r['group'] for r in groups]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean complete callback time (ms)');axes[0].set_ylim(0,310)
    axes[1].set_ylabel('Callbacks exceeding 500 ms (%)');axes[1].set_ylim(0,16)
    fig.suptitle('Complete timing audit: 3,456 matched pairs; all failures retained',y=.98,fontsize=13)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.91),frameon=False)
    fig.text(.5,.035,'79,399 callbacks per implementation. Total measured time falls 76.04%; 2,299 buffered calls still exceed 500 ms.\n'
        'Mixed execution epochs and concurrent desktop load; no homogeneous-load or hard-real-time claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.84))
    for suffix in ('png','svg'):fig.savefig(OUT/f'complete_timing.{suffix}')
    plt.close(fig)
    plants=['nominal','gain_drag','lag_mild','lag_strong'];names=['Nominal','Gain / drag','Mild lag','Strong lag']
    lookup={(c['plant'],c['model_kind']):c for c in online['cells']}
    fig,axes=plt.subplots(1,3,figsize=(14,4.7))
    x=np.arange(4)
    for ki,kind in enumerate(('nominal','reference')):
        cells=[lookup[p,kind] for p in plants]
        for ai,field in enumerate(('cost_gain_percent','adopted','harmful_and_adopted')):
            values=[c[field] for c in cells]
            bars=axes[ai].bar(x+(ki-.5)*width,values,width,color=colors[ki],label=kind.title()+' prediction')
            labels=([f'{v:.3f}' for v in values] if ai==0 else
                [str(v) for v in values] if ai==1 else [f"{c[field]}/{c['adopted']}" for c in cells])
            axes[ai].bar_label(bars,labels=labels,padding=3+ki*8 if ai==0 else 3,fontsize=8)
            axes[ai].set_xticks(x,names,rotation=18);axes[ai].grid(axis='y',alpha=.15);axes[ai].set_axisbelow(True)
    axes[0].set_ylabel('Mean team-cost gain over baseline (%)');axes[0].set_ylim(0,.49)
    axes[1].set_ylabel('Episodes adopting learning / 15');axes[1].set_ylim(0,9)
    axes[2].set_ylabel('Harmful adopted episodes');axes[2].set_ylim(0,1.5);axes[2].set_yticks([0,1])
    fig.suptitle('Online actuator integration: complete 120-run comparison',y=.98,fontsize=13)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.91),frameon=False)
    fig.text(.5,.035,'Five old initials shared across plants and three trained models; each plant / variant contains 15 runs.\n'
        'All runs complete. Right-panel labels are harmed / adopted counts, not certified probabilities. No new independent efficacy samples.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.15,1,.84))
    for suffix in ('png','svg'):fig.savefig(OUT/f'online_integration.{suffix}')
    plt.close(fig)
    summary=dict(buffered_report_sha256=sha(BUFFER/'analysis/report.json'),online_report_sha256=sha(ONLINE/'analysis/report.json'),
        timing_groups=groups,online_cells=online['cells'],new_independent_initials=0,new_risk_certificate=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    manifest={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='sha256.json'}
    (OUT/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(groups=3,online_cells=8,figures=2,summary=summary['timing_groups'][0])))


if __name__=='__main__':main()
