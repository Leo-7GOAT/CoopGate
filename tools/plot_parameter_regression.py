"""Scientific case figure from reviewed actual SUMO,not population evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def plot():
    source=ROOT/'artifacts/mappo_ctde_20261101/development/parameter_scenario_regression_20261102'
    if not json.loads((source/'verification.json').read_text())['all_match']:raise ValueError('Reviewed actual SUMO required')
    rows=[json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines()]
    chosen={r['key'][-1]:r for r in rows if r['row']['disturbance']=='clean'}
    base=chosen['baseline']['row'];old=chosen['original_4705'];new=chosen['parameter_4705']
    old_delta=np.array(old['row']['per_vehicle_cost'])-base['per_vehicle_cost']
    new_delta=np.array(new['row']['per_vehicle_cost'])-base['per_vehicle_cost']
    trace=old['trace'];clock=np.arange(len(trace))*.5
    midpoint=np.array([t['reconstructed_actuator'][1] for t in trace])
    actual=np.array([0.]+[t['actuator_states'][-1][1] for t in trace[:-1]])
    lower=np.array([t['actuator_memory_lower'][1] for t in trace]);upper=np.array([t['actuator_memory_upper'][1] for t in trace])
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    x=np.arange(3);axes[0].bar(x-.19,old_delta,.38,label='Original 33-path gate',color='#d55e00')
    axes[0].bar(x+.19,new_delta,.38,label='Parameter-scenario gate',color='#0072b2')
    axes[0].axhline(0,color='black',lw=.7);axes[0].set_xticks(x,['Front','Ramp','Rear'])
    axes[0].set_ylabel('Cost minus common baseline (s)');axes[0].set_title('Actual closed-loop outcome')
    axes[0].legend(frameon=False,fontsize=8);axes[0].grid(axis='y',alpha=.2)
    axes[1].fill_between(clock,lower,upper,color='#999999',alpha=.25,label='Online memory interval')
    axes[1].plot(clock,midpoint,color='#d55e00',ls='--',label='Midpoint reconstruction')
    axes[1].plot(clock,actual,color='black',lw=1.5,label='True state (offline only)')
    axes[1].set_xlabel('Decision start time (s)');axes[1].set_ylabel('Ramp actuator memory (m/s²)')
    axes[1].set_title('Original gate: uncertainty versus point model');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Known regression fixture: slow yield, strong lag, clean mode',fontsize=11)
    fig.supxlabel('One reused initial; selected-case mechanism illustration, no population inference.',fontsize=8)
    out=ROOT/'artifacts/mappo_ctde_20261101/parameter_regression_figure_20261102';out.mkdir(exist_ok=False)
    fig.savefig(out/'mechanism.png',dpi=180);fig.savefig(out/'mechanism.svg');plt.close(fig)
    (out/'data.json').write_text(json.dumps(dict(case=[96902001,'slow_yield','lag_strong','clean'],
        old_delta_cost=old_delta.tolist(),new_delta_cost=new_delta.tolist(),clock=clock.tolist(),
        midpoint=midpoint.tolist(),true_memory_offline=actual.tolist(),lower=lower.tolist(),upper=upper.tolist(),
        independent_initials=1,extra_sumo_runs=0),indent=2)+'\n')
    print(str(out/'mechanism.png'),flush=True)


if __name__=='__main__':plot()
