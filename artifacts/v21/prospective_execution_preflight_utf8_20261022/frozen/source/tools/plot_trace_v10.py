"""Plot the predesignated first dense/bias trace, without outcome selection."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v10'
protocol=json.loads((D/'protocol.json').read_text());seed=protocol['test_seeds']['dense'][0]
initial=next(r['state'] for r in json.loads((D/'study/initial_states.json').read_text()) if r['seed']==seed)
rows=list(csv.DictReader((D/'study/episodes.csv').open()))
fig,axes=plt.subplots(3,2,figsize=(11,8),sharex=True,sharey='row',layout='constrained')
for col,(method,title) in enumerate((('v9_learned_pair','V9 learned pair'),('rollout_learned','V10 coordinate rollout'))):
    trace=json.loads((D/f'study/trace_dense_bias_{method}.json').read_text())
    t=np.array([0.,*[r['t'] for r in trace]])
    x=np.array([initial,*[r['x'] for r in trace]])
    u=np.array([r['command'] for r in trace])
    for car in range(3):
        color=('#286d9b','#d67b2a','#247a61')[car]
        axes[0,col].plot(t,x[:,car,0],label=f'Car {car}',color=color)
        axes[1,col].plot(t,x[:,car,1],color=color)
        axes[2,col].step(t,np.r_[u[:,car],u[-1,car]],where='post',color=color)
    row=next(r for r in rows if int(r['seed'])==seed and r['disturbance']=='bias' and r['method']==method)
    axes[0,col].set_title(f"{title} | mean arrival {float(row['mean_capped_finish_s']):.3f} s")
    axes[0,col].axhspan(-25,6.8,color='#ccc',alpha=.2);axes[0,col].axhline(60,color='#777',ls='--',lw=.8)
    axes[0,col].legend(loc='upper left',fontsize=8)
    axes[2,col].set_xlabel('Simulation time (s)')
    for row_index in range(3):axes[row_index,col].grid(alpha=.18)
for row_index,label in enumerate(('Route position (m)','Speed (m/s)','Command (m/s²)')):axes[row_index,0].set_ylabel(label)
fig.suptitle(f'Predesignated dense scene {seed}, persistent actuator bias\nSame disturbance tape and robust safety certificate',fontsize=12)
fig.savefig(D/'trajectory_example.png',dpi=180);plt.close(fig)
