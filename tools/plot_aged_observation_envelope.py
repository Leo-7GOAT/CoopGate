"""One declared baseline trajectory illustration from the full envelope audit."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import audit_aged_observation_envelope as audit
from coopgate.episode_store_v21 import inventory, observations

ROOT = audit.ROOT
OUT = audit.OUT.parent/'aged_observation_figures'


def plot():
    if OUT.exists():
        raise FileExistsError('Preserve the first envelope figure')
    report = audit.read(audit.OUT/'report.json')
    proof = audit.read(audit.OUT/'verification.json')
    assert proof['audit_exit_code']==0 and proof['all_791_frames_checked']
    assert proof['report_sha256']==audit.digest(audit.OUT/'report.json')
    directory = audit.PARENT/'00002'
    manifest = audit.read(audit.PARENT/'manifest.json')
    assert audit.digest(directory/'observations.sqlite')==manifest['files_sha256']['00002/observations.sqlite']
    identity = inventory(directory/'observations.sqlite')
    cases = list(observations(directory/'observations.sqlite', identity['metadata'], identity['cases']))
    selected = [(key,body,trace) for key,body,trace in cases if key[-2:]==('baseline','none')]
    assert len(selected)==1
    key,body,trace = selected[0]
    rows = [r for r in report['rows'] if tuple(r['case'])==key]
    frames = audit.public_frames(trace,body['row']['terminal_attempt'])
    assert len(rows)==len(frames)
    times = np.array([r['decision']*.5 for r in rows])
    truth = np.array([r['true_state'] for r in rows])
    raw = np.array([r['observed_state'] for r in frames])
    lower = np.array([r['estimate']['lower'] for r in rows])
    upper = np.array([r['estimate']['upper'] for r in rows])
    figure,axes = plt.subplots(3,1,figsize=(9.2,8.6),sharex=True)
    for actor,axis in enumerate(axes):
        axis.fill_between(times,lower[:,actor,0],upper[:,actor,0],color='#2676ad',alpha=.2,
                          label='Estimated current-state interval')
        axis.plot(times,truth[:,actor,0],color='#202020',lw=2,label='True current position (audit only)')
        axis.plot(times,(lower[:,actor,0]+upper[:,actor,0])/2,color='#2676ad',lw=1.4,ls='--',
                  label='Interval midpoint')
        axis.plot(times,raw[:,actor,0],color='#d46c20',lw=1.2,marker='.',ms=4,
                  label='Received stale position')
        axis.set_ylabel(f'Vehicle {actor+1}\nPosition (m)')
        axis.grid(alpha=.2)
    axes[-1].set_xlabel('Current physical decision time (s)')
    axes[0].set_title('Severe / stale-dropout development baseline: received sample versus current-state envelope',fontsize=10)
    handles,labels = axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9)
    figure.tight_layout(rect=(0,.09,1,1))
    OUT.mkdir(parents=True,exist_ok=False)
    figure.savefig(OUT/'stale_baseline_positions.png',dpi=180)
    plt.close(figure)
    receipt = dict(report_sha256=proof['report_sha256'],source_sha256=audit.digest(Path(__file__)),
        database_sha256=audit.digest(directory/'observations.sqlite'),illustrated_case=list(key),
        received_frames=len(rows),figure_sha256=audit.digest(OUT/'stale_baseline_positions.png'),
        scope='One predetermined severe/stale baseline illustration, including its terminal received frame. Full audit retains all 39 trajectories.',
        new_sumo_executions=0,original_task_failure_unchanged=True,new_control_performance_claimed=False)
    with (OUT/'figure_receipt.json').open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(receipt,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    plot()
