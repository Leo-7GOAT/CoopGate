"""Export only after all 3,200 factorial runs and original full audit finish."""
import json

import numpy as np

from coopgate.episode_store_v21 import observations
from coopgate.joint_risk_data_v42 import episode_outcome
import design_cspi_factorial_comparison as design
import execute_candidate_factorial as parent


def export():
    supplement=design.inputs();plan=parent.inputs();sha=parent.io.digest(parent.PLAN)
    manifest=parent.terminal(sha) # Refuses incomplete/failed/live native execution.
    analysis=parent.OUT/'analysis'
    for name in ('report.json','replay.json','verification.json'):parent.io.committed(analysis/name)
    proof=parent.io.read(analysis/'verification.json')
    assert proof['plan_sha256']==sha and proof['actual_sumo_runs']==3200
    assert proof['report_sha256']==parent.io.digest(analysis/'report.json')
    assert proof['replay_sha256']==parent.io.digest(analysis/'replay.json')
    assert len(parent.io.read(analysis/'replay.json')['reviews'])==3200
    before={name:parent.io.digest(parent.OUT/'execution'/name) for name in (*parent.TERMINAL_FILES,'exit.json')}
    assert before==proof['terminal_hashes']
    features=[];team=[];costs=[];arrivals=[];failed=[];adopted=[]
    for assignment in supplement['assignments']:
        case=plan['cases'][assignment['block_index']]
        assert assignment['cluster_id']==case['cluster_id']
        directory=parent.OUT/'blocks'/f"{case['block_index']:05d}"
        meta=parent.metadata(case,sha);rows={}
        for key,body,_ in observations(directory/'observations.sqlite',meta,parent.schedule(case)):
            assert key==[case['cluster_id'],body['variant']] and body['variant'] not in rows
            rows[body['variant']]=body['row']
        assert set(rows)==set(parent.VARIANTS)
        state=np.asarray(rows['baseline']['initial'])
        assert state.shape==(3,2)
        outcomes=[episode_outcome(rows[v]) for v in design.VARIANTS]
        assert not outcomes[0]['adopted']
        features.append(state.ravel().tolist()+[float(case['plant']==p) for p in design.PLANTS]
            +[float(case['policy_id']==p) for p in design.MODELS])
        team.append([o['team_cost'] for o in outcomes]);costs.append([o['costs'] for o in outcomes])
        arrivals.append([o['arrivals'] for o in outcomes]);failed.append([o['failed'] for o in outcomes])
        adopted.append([o['adopted'] for o in outcomes])
    assert manifest==parent.terminal(sha) and design.inputs()==supplement
    assert before=={name:parent.io.digest(parent.OUT/'execution'/name) for name in before}
    data=design.OUT/'data';data.mkdir(exist_ok=False)
    features=np.asarray(features);team=np.asarray(team);failed=np.asarray(failed);treatment=np.array([a['treatment'] for a in supplement['assignments']])
    observed=-(team[np.arange(640),treatment]+25*failed[np.arange(640),treatment])
    for split in ('fit','tune','safety_test','evaluation'):
        mask=np.array([a['split']==split for a in supplement['assignments']])
        if split=='evaluation':np.savez_compressed(data/'evaluation_features.npz',features=features[mask])
        else:np.savez_compressed(data/(split+'.npz'),features=features[mask],treatment=treatment[mask],outcome=observed[mask])
    np.savez_compressed(data/'paired_audit_potential.npz',team_cost=team[512:],costs=np.asarray(costs)[512:],
        arrivals=np.asarray(arrivals)[512:],failed=failed[512:],adopted=np.asarray(adopted)[512:])
    design.write(data/'complete.json',dict(supplement_plan_sha256=design.digest(design.PLAN),
        parent_plan_sha256=sha,parent_full_audit_sha256=design.digest(analysis/'verification.json'),
        parent_runs_verified=3200,additional_sumo_runs=0,observed_rows=512,evaluation_rows=128,
        files_sha256={p.name:design.digest(p) for p in data.iterdir() if p.is_file()}))
    print(json.dumps(dict(parent_runs_verified=3200,observed_rows=512,evaluation_rows=128)))


if __name__=='__main__':export()
