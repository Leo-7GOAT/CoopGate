"""Fixed example rule: first dense initial state, persistent bias, both versions."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v11'


def main():
    protocol=json.loads((D/'protocol.json').read_text());seed=protocol['test_seeds']['dense'][0]
    initial=next(r['state'] for r in json.loads((D/'study/initial_states.json').read_text()) if r['seed']==seed)
    rows=list(csv.DictReader((D/'study/episodes.csv').open()))
    fig,axes=plt.subplots(4,2,figsize=(11,10),sharex=True,sharey='row',layout='constrained')
    for col,(method,title) in enumerate((('v10_rollout','V10 arrival-only rollout'),('smooth_learned','V11 lexicographic rollout'))):
        trace=json.loads((D/f'study/trace_dense_bias_{method}.json').read_text())
        times=np.array([0.,*[r['t'] for r in trace]])
        x=np.array([initial,*[r['x'] for r in trace]])
        u=np.array([r['command'] for r in trace])
        for car,color in enumerate(('#286d9b','#d67b2a','#247a61')):
            axes[0,col].plot(times,x[:,car,0],color=color,label=f'Car {car}')
            axes[1,col].plot(times,x[:,car,1],color=color)
            # Trace t is the end of the commanded interval, not its start.
            axes[2,col].step(times,np.r_[u[:,car],u[-1,car]],where='post',color=color)
        variation=np.abs(np.diff(np.vstack((np.zeros(3),u)),axis=0)).sum(axis=1).cumsum()
        axes[3,col].step(np.r_[0.,times[:-1],times[-1]],np.r_[0.,variation,variation[-1]],where='post',color='#40474c')
        row=next(r for r in rows if int(r['seed'])==seed and r['disturbance']=='bias' and r['method']==method)
        axes[0,col].set_title(f"{title}\nArrival {float(row['mean_capped_finish_s']):.3f} s | TV {float(row['total_command_variation']):.3f} m/s²")
        axes[0,col].axhspan(-25,6.8,color='#ccc',alpha=.2)
        axes[0,col].axhline(60,color='#777',ls='--',lw=.8)
        axes[0,col].legend(loc='upper left',fontsize=8)
        axes[3,col].set_xlabel('Simulation time (s)')
        for ax in axes[:,col]:ax.grid(alpha=.18)
    for ax,label in zip(axes[:,0],('Route position (m)','Speed (m/s)','Command (m/s²)','Cumulative total\nvariation (m/s²)')):ax.set_ylabel(label)
    fig.suptitle(f'Fixed example: first dense scene {seed}, persistent actuator bias\nSame initial state, disturbance tape and joint safety certificate',fontsize=12)
    fig.savefig(D/'trajectory_example.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
