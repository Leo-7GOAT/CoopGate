"""Verify complete fixed deployment evidence, preserving every failed episode."""
import json
from pathlib import Path

import numpy as np

import run_dmps_nominal_model_development as run


def verify(output):
    done=run.io.read(output/'complete.json');plan=run.io.read(output/'plan.json')
    assert done['plan_sha256']==run.io.digest(output/'plan.json')
    assert plan['cases']==run.schedule()==[x['case'] for x in done['cases']]
    assert not plan['privileged_forecasts'] and not plan['comparative_efficacy_claimed']
    assert run.validate_parent()[1]==plan['parent_complete_sha256']
    assert run.io.digest(run.PARENT/'trained_state.pt')==plan['checkpoint_sha256']
    for name,expected in plan['source_sha256'].items():
        assert run.io.digest(run.ROOT/name)==expected
    cases=[];times=[];interventions=0
    for item in done['cases']:
        folder=output/f"{item['case']['index']:03d}"
        assert run.io.digest(folder/'complete.json')==item['complete_sha256']
        receipt=run.io.read(folder/'complete.json');assert receipt['result']==item['result']
        for name,expected in receipt['files_sha256'].items():assert run.io.digest(folder/name)==expected
        if 'reset_failure' in item['result']:
            cases.append(item['result']);continue
        record=run.io.read(folder/'episode.json');review=run.audit_episode(record)
        assert review==run.io.read(folder/'review.json')
        assert dict(case=item['case'],**review)==item['result']
        for decision in record['decisions']:
            assert run.io.read(folder/f"decision_{decision['step']:03d}.json")==decision
            times.append(decision['wall_ms']);interventions+=decision['planner']['intervention']
        cases.append(dict(case=item['case'],**{k:v for k,v in review.items() if k!='prediction_errors'},
            final_state=record['decisions'][-1]['trace']['x'] if record['decisions'] else record['initial'],
            truncated=record['truncated']))
    return dict(complete_sha256=run.io.digest(output/'complete.json'),cases=cases,
        actual_decisions=sum(r.get('actual_decisions',0) for r in cases),
        physical_substeps=sum(r.get('physical_substeps',0) for r in cases),
        completed_episodes=sum(r['completed'] for r in cases),original_planner_interventions=interventions,
        nominal_model_decisions=sum(r.get('nominal_model_decisions',0) for r in cases),
        wall_ms=dict(mean=float(np.mean(times)),p95=float(np.quantile(times,.95)),maximum=max(times)),
        independent_physics_and_action_audit=True,source_and_checkpoint_hashes_verified=True,
        comparative_efficacy_claimed=False,new_risk_certificate=False)


if __name__=='__main__':
    output=run.ROOT/'artifacts/literature_reproduction/dmps_nominal_model_development'
    result=verify(output)
    run.io.exclusive_json(output/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
