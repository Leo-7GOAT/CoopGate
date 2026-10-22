"""Plot one explicitly selected SUMO interaction counterexample."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(folder):
    folder=Path(folder);p=json.loads((folder/'protocol.json').read_text());case=p['cases'][0]
    rows=json.loads((folder/'episodes.json').read_text())
    styles=[('all_initial','Single learner + rule neighbors'),('full','Complete learned joint policy'),
            ('restore_1','Pretrained ego spliced into learned team'),('team_planner','Strong cooperative planner')]
    fig,axes=plt.subplots(2,2,figsize=(10.4,6.8),sharex=True,sharey=True,layout='constrained')
    colors=['#3177a8','#d16e27','#198978']
    for ax,(name,title) in zip(axes.flat,styles):
        trace=json.loads((folder/f'trace_0_{name}.json').read_text())
        row=next(r for r in rows if r['seed']==case['seed'] and r['disturbance']==case['disturbance'] and r['controller']==name)
        times=np.array([0.]+[r['t'] for r in trace]);positions=np.array([row['initial']]+[r['x'] for r in trace])[:,:,0]
        ax.axhspan(-25,6.8,color='#f5e5bf',alpha=.7,label='Conflict region')
        ax.axhline(60,color='#777',linestyle='--',linewidth=.9)
        for i in range(3):ax.plot(times,positions[:,i],color=colors[i],linewidth=1.7,label=['Mainroad 0','Ramp 1','Mainroad 2'][i])
        ax.set_title(title,fontsize=10);ax.set_xlim(0,25);ax.set_ylim(-155,90);ax.grid(alpha=.15)
        ax.text(.03,.94,'Completed' if row['completed'] else 'Timed out at 25 s',transform=ax.transAxes,va='top',
            fontsize=9,color='#257264' if row['completed'] else '#af3428')
        ax.set_xlabel('Time (s)');ax.set_ylabel('Route coordinate (m)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle(f"Selected SUMO counterexample: seed {case['seed']}, {case['disturbance']} disturbance\nBoth complete policies succeed; their mixed actors can stall",fontsize=11)
    fig.savefig(folder/'interaction.png',dpi=180);fig.savefig(folder/'interaction.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();plot(a.folder)
