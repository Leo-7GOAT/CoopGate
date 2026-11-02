from pathlib import Path
import csv
import json
import platform
from importlib.metadata import version

import numpy as np
import torch

from .config import Config
from .mappo import load_policy
from .shield import Supervisor
from .sim import SumoEnv


def wilson(successes, trials, z=1.96):
    if trials == 0:
        return [0.0, 1.0]
    p=successes/trials
    d=1+z*z/trials
    center=(p+z*z/(2*trials))/d
    half=z*np.sqrt(p*(1-p)/trials+z*z/(4*trials**2))/d
    return [float(max(0,center-half)),float(min(1,center+half))]


def summarize(rows):
    output={}
    for method in sorted({r['method'] for r in rows}):
        subset=[r for r in rows if r['method']==method]
        n=len(subset)
        coll=sum(r['collision'] for r in subset)
        succ=sum(r['success'] for r in subset)
        output[method]={'episodes':n,'collision_rate':coll/n,'collision_wilson95':wilson(coll,n),
                        'success_rate':succ/n,'success_wilson95':wilson(succ,n)}
        for metric in ('elapsed_s','completion_fraction','progress_m','rl_fraction','switches',
                       'emergency_steps','gate_mean_ms','gate_p95_ms','checks_per_step','mean_abs_jerk'):
            output[method][metric]=float(np.mean([r[metric] for r in subset]))
        successful=[r['elapsed_s'] for r in subset if r['success']]
        output[method]['successful_time_s']=float(np.mean(successful)) if successful else None
    return output


def evaluate(checkpoint, out, seeds=10, start_seed=10000, methods=None,
             scenarios=('nominal','dense','braking'), cfg=None, gui=False):
    cfg=cfg or Config()
    model,metadata=load_policy(checkpoint)
    # Preserve the v1 command's default suite; v2 requires explicit executor flags.
    methods=methods or tuple(m for m in Supervisor.methods if not m.startswith('paired'))
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for method in methods:
        with SumoEnv(cfg,method,gui=gui) as env:
            for scenario_index, scenario in enumerate(scenarios):
                for k in range(seeds):
                    obs=env.reset(start_seed+1000*scenario_index+k,scenario)
                    done=False
                    while not done:
                        action,_,_=model.act(obs,deterministic=True)
                        obs,_,terminal,truncated,info=env.step(action)
                        done=terminal or truncated
                    rows.append(info)
                    if k == 0:
                        (out/f'trace_{method}_{scenario}.json').write_text(json.dumps(env.trace))
                print(json.dumps({'method':method,'scenario':scenario,'completed_episodes':len(rows)}),flush=True)
        with (out/'episodes.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (out/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
    manifest={'python':platform.python_version(),'torch':torch.__version__,'sumo':version('eclipse-sumo'),
              'training_seed':metadata['seed'],'training_steps':metadata['steps'],
              'eval_seed_start':start_seed,'seeds_per_scenario':seeds,
              'scenarios':list(scenarios),'methods':list(methods),'config':cfg.dict(),
              'claims':'Pilot only. One training seed per manifest. Nominal rollout is not a formal safety certificate.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    plot_results(rows,out)
    return rows


def plot_results(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=summarize(rows)
    methods=list(summary)
    labels=[m.replace('_','\n') for m in methods]
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric,title in zip(axs,('collision_rate','success_rate','rl_fraction'),
                              ('Collision rate (lower is better)','Task success rate','RL-mode fraction')):
        values=[summary[m][metric] for m in methods]
        ax.bar(labels,values,color=['#64748b','#eab308','#0891b2','#7c3aed','#ea580c','#16a34a'])
        ax.set_ylim(0,1.05); ax.set_title(title); ax.grid(axis='y',alpha=.2)
        if metric in ('collision_rate','success_rate'):
            key='collision_wilson95' if metric=='collision_rate' else 'success_wilson95'
            errors=np.array([[max(0,values[i]-summary[m][key][0]),max(0,summary[m][key][1]-values[i])] for i,m in enumerate(methods)]).T
            ax.errorbar(range(len(methods)),values,yerr=errors,fmt='none',c='black',capsize=3)
        ax.tick_params(axis='x',labelsize=8)
    fig.suptitle('CoopGate / real SUMO pilot — episode-level Wilson 95% intervals')
    fig.savefig(Path(out)/'comparison.png',dpi=170)
    fig.savefig(Path(out)/'comparison.pdf')
    plt.close(fig)


def render_trace(trace_path, destination):
    """Render a saved SUMO trajectory; this is a replay, not simulator pixels."""
    from PIL import Image,ImageDraw
    records=json.loads(Path(trace_path).read_text())
    frames=[]
    colors=['#38bdf8','#34d399','#fbbf24']
    for record in records:
        img=Image.new('RGB',(1100,310),'#0f172a')
        d=ImageDraw.Draw(img)
        d.rectangle((25,120,1075,215),fill='#334155')
        merge=25+220/450*1050
        d.polygon([(merge,170),(merge+55,215),(1075,215),(1075,170)],fill='#0f172a')
        for sx in range(30,int(merge),35):
            d.line((sx,170,sx+18,170),fill='#cbd5e1',width=2)
        d.text((25,20),'CoopGate | SUMO trajectory replay (not a camera recording)',fill='white')
        d.text((25,47),f"t={record['t']:.1f}s   modes={record['modes']}   rollout score={record['score']:.3f}",fill='#cbd5e1')
        for i,s in enumerate(record['states']):
            if not s[4] or s[0]>450:
                continue
            px=25+s[0]/450*1050; py=193-s[1]/3.2*46
            color=colors[i] if i<3 else '#94a3b8'
            d.rounded_rectangle((px-12,py-9,px+12,py+9),radius=3,fill=color)
            d.text((px-15,py-29),str(i),fill='white')
        d.text((25,260),'Controlled vehicles: 0, 1, 2 | Background: 3, 4 | 1=RL, 0=backup',fill='#cbd5e1')
        frames.append(img)
    if not frames:
        raise ValueError('Empty trace')
    frames[0].save(destination,save_all=True,append_images=frames[1:],duration=150,loop=0)
