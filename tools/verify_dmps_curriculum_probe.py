"""Post-exit native reset, full substep and fixed-case audit; no efficacy claim."""
import json

import numpy as np

from coopgate.dmps_curriculum_initials import target_state
import probe_dmps_curriculum_sumo as probe

OUT=probe.ROOT/'artifacts/literature_reproduction/dmps_curriculum_native_probe'


def verify(output=OUT):
    io=probe.base.io
    done=io.read(output/'complete.json');plan=io.read(output/'plan.json')
    receipt=io.read(output/'observed_execution_exit.json')
    assert receipt['exit_code']==0 and receipt['complete_sha256']==io.digest(output/'complete.json')
    assert done['plan_sha256']==io.digest(output/'plan.json')
    assert plan['cases']==probe.cases()==[e['case'] for e in done['episodes']]
    for name,expected in plan['source_sha256'].items():assert io.digest(probe.ROOT/name)==expected
    for name,expected in done['files_sha256'].items():assert io.digest(output/name)==expected
    cases=[];decisions=substeps=0
    for index,entry in enumerate(done['episodes']):
        folder=output/'execution'/f'episode_{index:03d}'
        record=io.read(folder/'episode.json');initial=io.read(folder/'initialization.json')
        assert record['case']==entry['case'] and initial['phase']==entry['case']['phase']
        assert not initial['placement_after_time_origin']
        np.testing.assert_array_equal(record['initial'],initial['actual_initial'])
        expected=target_state(entry['case']['seed'],entry['case']['phase'])
        if expected is None:assert initial['target_state'] is None
        else:
            np.testing.assert_array_equal(initial['target_state'],expected)
            np.testing.assert_allclose(record['initial'],expected,atol=1e-8,rtol=0)
        assert initial['starting_actual_decisions']==decisions
        review=probe.base.evidence.audit_episode(record)
        assert review==entry['review']==io.read(folder/'review.json')
        assert io.digest(folder/'episode.json')==entry['episode_sha256']
        for item in record['decisions']:
            assert io.read(folder/f"decision_{item['step']:03d}.json")==item
        assert not review['collision'] and review['geometry_violations']==0
        if entry['case']['phase']==0:assert review['completed']
        decisions+=review['actual_decisions'];substeps+=review['physical_substeps']
        cases.append(dict(case=entry['case'],completed=review['completed'],
            actual_decisions=review['actual_decisions'],physical_substeps=review['physical_substeps'],
            initialization=initial))
    assert decisions==done['actual_decisions']
    assert len(list((output/'execution').glob('call_*.json')))==decisions
    return dict(complete_sha256=io.digest(output/'complete.json'),observed_exit=receipt,
        cases=cases,native_initials_verified=8,actual_decisions=decisions,physical_substeps=substeps,
        near_goal_team_completions=sum(c['completed'] for c in cases if c['case']['phase']==0),
        all_physics_and_model_predictions_reconstructed=True,training_steps=0,
        comparative_efficacy_claimed=False,new_risk_certificate=False)


if __name__=='__main__':
    result=verify();probe.base.io.exclusive_json(OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
