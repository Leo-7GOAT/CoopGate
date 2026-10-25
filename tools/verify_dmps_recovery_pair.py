"""Post-exit replay and descriptive utility summary of every paired condition."""
import json
from pathlib import Path

import numpy as np

import pair_dmps_recovery_baseline as study

OUT=study.ROOT/'artifacts/literature_reproduction/dmps_recovery_quality'


def verify(output=OUT):
    io=study.parent.base.io
    done=io.read(output/'complete.json');plan=io.read(output/'plan.json')
    receipt=io.read(output/'observed_execution_exit.json')
    assert receipt['exit_code']==0 and receipt['complete_sha256']==io.digest(output/'complete.json')
    assert done['plan_sha256']==io.digest(output/'plan.json')
    assert plan['cases']==study.parent.evaluation_cases()==[e['case'] for e in done['cases']]
    _,parent=study.prerequisites()
    assert plan['parent_verification_sha256']==io.digest(study.PARENT/'verification.json')
    for name,expected in plan['source_sha256'].items():assert io.digest(study.ROOT/name)==expected
    results=[];substeps=commands=0
    for item,learned in zip(done['cases'],parent['evaluation']):
        expected=item['case'];folder=output/f"{expected['index']:03d}"
        assert io.digest(folder/'complete.json')==item['complete_sha256']
        sealed=io.read(folder/'complete.json');assert sealed['result']==item['result']
        for name,digest in sealed['files_sha256'].items():assert io.digest(folder/name)==digest
        result=item['result'];assert result['learning_review']==learned['review']
        source_folder=study.PARENT/'evaluation'/f"episode_{expected['index']:03d}"
        source=io.read(source_folder/'episode.json') if (source_folder/'episode.json').exists() else None
        if result['baseline_row'] is None:
            assert result['baseline_failure'] and (folder/'failure.json').exists()
            assert result['baseline_failed'] and result['relative_harm'] is None
            assert result['paired_cost_gain_percent'] is None
        else:
            record=io.read(folder/'episode.json');assert record['row']==result['baseline_row']
            assert record['row']['method']=='bounded_actuator_recovery_baseline'
            case=dict(expected,cluster_id=f"dmps_quality_{expected['index']:03d}",scenario='nominal')
            case.update(initial_sha256=study.baseline.design.shared.array_sha256(np.asarray(record['row']['initial']),decimals=8),
                dynamics_sha256=study.baseline.design.shared.canonical_sha256(study.matched_plant_parameters(case['seed'],case['plant'])),
                disturbance_sha256=study.baseline.design.shared.array_sha256(study.disturbance_tape(case['seed'],case['disturbance'],.1,250)))
            if source is not None:np.testing.assert_allclose(record['row']['initial'],source['initial'],atol=1e-8,rtol=0)
            review=study.baseline.audit_record(case,record,None)
            assert review==io.read(folder/'review.json')==result['baseline_review']
            outcome=study.paired_outcome(record['row'],learned['review'],source['capped_finish'] if source else None)
            assert all(result[k]==v for k,v in outcome.items())
            substeps+=review['physical_substeps'];commands+=review['reference_replayed_commands']
        results.append(result)
    complete=all(r['relative_performance_assessable'] for r in results)
    before=sum(r['baseline_row']['team_cost'] for r in results) if complete else None
    after=sum(float(np.mean(r['learning_review']['per_vehicle_cost'])) for r in results) if complete else None
    return dict(complete_sha256=io.digest(output/'complete.json'),observed_exit=receipt,
        actual_baseline_conditions=10,baseline_completed=sum(not r['baseline_failed'] for r in results),
        learned_completed=sum(not r['learned_failed'] for r in results),
        assessable_pairs=sum(r['relative_performance_assessable'] for r in results),
        harms_with_complete_baseline=sum(r['relative_harm'] is True for r in results),
        unassessable_baseline_pairs=sum(r['baseline_failed'] for r in results),
        all_scope_cost_gain_percent=100*(before-after)/before if complete else None,
        baseline_physical_substeps_reconstructed=substeps,baseline_commands_replayed=commands,
        cases=results,independent_initials=2,new_risk_certificate=False,
        comparative_family_superiority_claimed=False,matched_training_budget_claimed=False,
        full_cohort_failures_retained=True)


if __name__=='__main__':
    result=verify()
    study.parent.base.io.exclusive_json(OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
