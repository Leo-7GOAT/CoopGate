"""Publication figure layout, separate from the frozen statistical computation."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v11'


def main():
    r=json.loads((D/'summary.json').read_text())
    selected=['v9_learned_pair','v10_rollout','smooth_time_only','smooth_binary','smooth_single_family','smooth_learned']
    labels=['V9\nlearned','V10\nrollout','Time-only\nobjective','Binary\ntail','Single\nfamily','V11\ncomplete']
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    for ax,field,ylabel in zip(axes,['mean_cost_s','mean_total_variation','mean_command_jerk'],
        ['Capped mean arrival time (s)','Command total variation (m/s²)','Mean command change rate (m/s³)']):
        values=[r['perturbed'][m][field] for m in selected]
        ax.bar(labels,values,color=['#99a4ad','#708d9b','#a9bfaf','#90b4a3','#6c9f88','#196955'])
        ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=8)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
        ax.set_ylim(0,max(values)*1.14)
    fig.suptitle('Same joint safety certificate | 600 perturbed runs per method | 200 independent initial states',fontsize=11)
    fig.savefig(D/'results.png',dpi=180);plt.close(fig)
    # The visual layout changes no statistics, protocol, or source hash.
    hashes={str(p.relative_to(D)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(D.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    (D/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n',newline='\n')
    print(f'Updated plot layout and sealed {len(hashes)} artifacts.')


if __name__=='__main__':main()
