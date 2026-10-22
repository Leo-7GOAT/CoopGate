"""Secondary predeclared selector analysis from paired SUMO counterfactuals.

No new SUMO execution is claimed: an episode-level selector picks one of the
already executed complete controllers on exactly the same initial/error tape.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from coopgate.confidence_v12 import conformal_upper

ROOT=Path(__file__).resolve().parents[1]


def plan(path):
    path=Path(path)
    if path.exists():raise RuntimeError('Analysis plan already exists')
    path.write_text(json.dumps({'alphas':[.05,.1,.2,.5],'uncalibrated':True,'minimum_gain_cost':.05,
        'analysis':'recombine paired complete SUMO trajectories; no additional physical executions',
        'primary_threshold_unchanged':.1,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


def run(study,out):
    plan_data=json.loads((ROOT/'artifacts/v12/single/gate_analysis_plan.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan_data['source_sha256']
    protocol=json.loads((ROOT/'artifacts/v12/single/protocol.json').read_text())
    rows=json.loads((Path(study)/'episodes.json').read_text());look={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    residuals=json.loads((ROOT/'artifacts/v12/single/calibration/residuals.json').read_text())
    score=lambda r:r['mean_capped_finish_s']+.05*r['variation']
    output=[]
    for label,alpha in [('uncalibrated',None)]+[(str(a),a) for a in plan_data['alphas']]:
        metrics=[]
        for r in rows:
            if not r['controller'].startswith('gate_'):continue
            seed=int(r['controller'].split('_')[1]);group=r['group']
            residual=[v['residual'] for v in residuals if v['policy']==seed and v['group']==group]
            q=0. if alpha is None else conformal_upper(residual,alpha)
            accepted=r['predicted_delta_cost']+q<=-.05
            baseline=look[(r['seed'],r['disturbance'],'rollout')];rl=look[(r['seed'],r['disturbance'],f'rl_{seed}')]
            delta=score(rl)-score(baseline)
            metrics.append({'accepted':accepted,'harmful':accepted and delta>1e-9,'cost_gain':-delta if accepted else 0.,
                'available_gain':-delta,'group':group})
        for group in ('all','low_conflict','high_conflict'):
            rr=[r for r in metrics if group=='all' or r['group']==group];n=len(rr);accept=sum(r['accepted'] for r in rr)
            output.append({'threshold':label,'group':group,'n':n,'accepted':accept,'acceptance_rate':accept/n,
                'harmful_rate':sum(r['harmful'] for r in rr)/n,'conditional_harm':sum(r['harmful'] for r in rr)/accept if accept else None,
                'mean_cost_gain':float(np.mean([r['cost_gain'] for r in rr]))})
    Path(out).write_text(json.dumps({'evaluation':'paired counterfactual recombination','rows':output},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--study');p.add_argument('--out');a=p.parse_args()
    plan(a.plan) if a.plan else run(a.study,a.out)
