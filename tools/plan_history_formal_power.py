"""Complete-pilot-only, fixed-grid descriptive bootstrap sizing projection."""
import json,time
from pathlib import Path
import numpy as np
from scipy.stats import beta,binom,t
from coopgate.episode_store_v21 import observations
import mappo_coupled_formal_experiment as e
from mappo_coupled_joint_data import episode_outcome

PILOT=e.BASE/'development/history_power_pilot_20261103'
OUT=e.BASE/'history_power_design_20261103'
B=2048;STREAM=550401;STRESS=.0005;ERROR=.05/30
GRID=sorted([(a,c) for a in (3072,4096,5120,6144) for c in (1024,1536,2048)],key=lambda pair:(7*pair[0]+16*pair[1],pair))


def simulate(adopt,harm,gain,plants,cal,conf,*,replicates=B):
    rng=np.random.default_rng(np.random.SeedSequence([STREAM,cal,conf]));counts=[];harms=[]
    for profile in e.design.PLANTS:
        ids=np.flatnonzero(plants==profile);index=rng.choice(ids,size=(replicates,cal//4))
        a=adopt[index];h=harm[index].copy();p=adopt[ids].mean(axis=0)
        shock=rng.random((replicates,cal//4,1))
        h|=a&(shock<np.minimum(1.,STRESS/np.maximum(p,1e-15)))
        counts.append(a.sum(axis=1));harms.append((a&h).sum(axis=1))
    a=np.array(counts);h=np.array(harms);n=cal//4;tail=ERROR/8
    upper=np.where(h<n,beta.isf(tail,h+1,n-h),1.)
    lower=np.where(a>0,beta.ppf(tail,a,n-a+1),0.)
    denominator=lower.mean(axis=0)
    mix=np.divide(upper.mean(axis=0),denominator,out=np.ones_like(denominator),where=denominator>0)
    strong=np.where(h[-1]<a[-1],beta.isf(ERROR,h[-1]+1,a[-1]-h[-1]),1.)
    risk_ok=np.all((mix<=.05-1e-12)&(strong<=.05-1e-12)&(binom.cdf(h[-1],a[-1],.05)<=ERROR),axis=1)
    index=rng.integers(len(adopt),size=(replicates,conf));ac=adopt[index].sum(axis=1);g=gain[index]
    alo=np.where(ac>0,beta.ppf(ERROR,ac,conf-ac+1),0.)
    adoption_ok=np.all((alo>=.2+1e-12)&(binom.sf(ac-1,conf,.2)<=ERROR),axis=1)
    mean=g.mean(axis=1);sd=g.std(axis=1,ddof=1);se=sd/np.sqrt(conf)
    gl=mean-t.isf(ERROR,conf-1)*se
    statistic=np.divide(mean-.5,se,out=np.zeros_like(mean),where=se>1e-12)
    gain_ok=np.all((sd>1e-12)&(gl>.5)&(t.sf(statistic,conf-1)<=ERROR),axis=1)
    passed=risk_ok&adoption_ok&gain_ok;k=int(passed.sum())
    mc_lower=float(beta.ppf(.01/len(GRID),k,replicates-k+1)) if k else 0.
    return dict(calibration_initials=cal,confirmation_initials=conf,actual_runs_if_registered=7*cal+16*conf,
        primary_runs_if_registered=4*(cal+conf),replicates=replicates,synthetic_joint_passes=k,
        synthetic_joint_pass_fraction=float(passed.mean()),mc_joint_pass_lower=mc_lower,
        synthetic_risk_pass_fraction=float(risk_ok.mean()),synthetic_adoption_pass_fraction=float(adoption_ok.mean()),
        synthetic_gain_pass_fraction=float(gain_ok.mean()),meets_planning_rule=mc_lower>.8)


def run():
    # Root terminal witness FIRST;no partial effects may be inspected.
    owner=e.read(PILOT/'owner_native_exit.json')
    if owner['actual_native_exit_code']!=0 or owner['report_sha256']!=e.sha(PILOT/'report.json'):
        raise ValueError('Completed actual pilot owner required')
    report=e.read(PILOT/'report.json');plan=e.read(PILOT/'plan.json')
    if not report['completed'] or report['actual_sumo_executions']!=1280 or report['new_independent_development_initials']!=320 or any(c['actual_native_exit_code']!=0 for c in report['children']):
        raise ValueError('Complete exact pilot scope/review required')
    if any(e.sha(e.ROOT/p)!=h for p,h in plan['sources'].items()):raise ValueError('Frozen pilot sources changed')
    arrays=[];failures=0
    for shard in range(4):
        summary=e.read(PILOT/f'shard{shard}/report.json')
        for block in summary['blocks']:
            db=PILOT/f'shard{shard}'/block['block']/'observations.sqlite'
            if e.sha(db)!=block['database_sha256']:raise ValueError('Pilot raw bytes changed')
            rows={r['method']:r['row'] for _,r,_ in observations(db,block['metadata'],block['keys'])}
            if set(rows)!=set(('baseline','4703','4704','4705')):raise ValueError('Full fixed-model row required')
            b=episode_outcome(rows['baseline']);aa=[];hh=[];gg=[];failures+=b['failed']
            for seed in ('4703','4704','4705'):
                q=episode_outcome(rows[seed]);aa.append(q['adopted']);failures+=q['failed']
                hh.append(bool(q['failed'] or b['failed'] or np.any(q['costs']>b['costs']+1e-9) or np.any(q['arrivals']>b['arrivals']+1e-9)))
                gg.append(100*(b['team_cost']-q['team_cost'])/b['team_cost'])
            arrays.append((block['metadata']['case']['plant'],aa,hh,gg))
    if len(arrays)!=320:raise ValueError('All320 complete pilot initials required')
    plants=np.array([r[0] for r in arrays]);adopt=np.array([r[1] for r in arrays],bool);harm=np.array([r[2] for r in arrays],bool);gain=np.array([r[3] for r in arrays],float)
    OUT.mkdir(exist_ok=False);started=time.perf_counter();results=[]
    for cal,conf in GRID:
        value=simulate(adopt,harm,gain,plants,cal,conf);results.append(value);print(json.dumps(value),flush=True)
    selected=next((r for r in results if r['meets_planning_rule']),None) if not failures else None
    final=dict(complete=True,pilot_initials=320,pilot_actual_sumo_runs=1280,pilot_execution_failures=int(failures),
        pilot_adopted_counts=adopt.sum(axis=0).tolist(),pilot_harmful_adopted_counts=(harm&adopt).sum(axis=0).tolist(),pilot_mean_gain_percent=gain.mean(axis=0).tolist(),
        fixed_grid=results,selected_design=selected,bootstrap_seed_stream=STREAM,extra_joint_harm_stress_per_class_model=STRESS,
        mc_tail_probability_per_design=.01/len(GRID),planning_joint_pass_lower_target=.8,
        assumptions='Empirical complete-row resampling plus declared common innovation harm shock;conditional planning only,not a guarantee of real power. No synthetic physical failures;actual pilot failure blocks selection.',
        new_error_allocated=False,new_formal_sumo_runs=0,formal_candidate_or_threshold_changed=False,
        pilot_report_sha256=e.sha(PILOT/'report.json'),source_sha256=e.sha(__file__),
        protocol_sha256=e.sha(e.ROOT/'docs/history_power_design_20261103_protocol.md'),elapsed_s=time.perf_counter()-started)
    e.write(OUT/'report.json',final);print(json.dumps(dict(selected=selected,pilot_failures=int(failures))),flush=True)


if __name__=='__main__':run()
