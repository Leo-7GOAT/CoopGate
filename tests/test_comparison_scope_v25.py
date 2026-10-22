from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_comparison_scope_v25 as audit
from coopgate.risk_comparators_v19 import conditional_ltt


def fixture():
    names=tuple(f'{p}_{s}' for p in ('raw','nominal','phase05') for s in audit.MODELS)
    adopted=np.ones((256,9),bool)
    harm=np.zeros_like(adopted)
    harm[:40,0]=True
    adopted[:,1]=False
    risk=conditional_ltt(range(256),names,adopted,harm,delta=.025)
    results={}
    for prefix in ('raw','nominal','phase05'):
        members={}
        for seed in audit.MODELS:
            members[seed]=dict(runs=20,selected_profiles={f'{prefix}_{seed}':20},
                actual_random_mode=dict(adopted=10,harmful_adopted=1,conditional_harm=.1,adoption_fraction=.5),
                worst_mode_bundle=dict(adopted=15,harmful_adopted=3,conditional_harm=.2,adoption_fraction=.75),
                mean_team_gain_percent=1.,mean_team_gain_seconds=.1,mean_team_gain_ci95=[.01,.2],
                completed=20,controller_p95_ms=600.)
        results[prefix]=dict(per_training_seed=members)
    return risk,deepcopy(risk),results


def test_all_models_same_original_levels_and_failure_and_zero_adoption_preserved():
    a,b,r=fixture()
    output=audit.shared_table(a,b,r,'phase05')
    assert len(output['rows'])==9
    assert output['per_candidate_target_test_level']==.025/9
    assert output['simultaneous_error_budget_both_targets']==.05
    assert output['whole_groups_under_shared_rule']==dict(raw=False,nominal=True,phase05=True)
    zero=next(row for row in output['rows'] if row['controller']=='raw' and row['model']=='1812')
    assert zero['calibration_actual']['empirical_conditional_risk'] is None
    assert zero['same_rule_both_targets_certified'] is False
    assert zero['final_actual_adoption']['conditional_harm']==.1


@pytest.mark.parametrize('kind',['delta','count','candidate','bound','certification','model','mixture','denominator'])
def test_reject_incomparable_or_altered_evidence(kind):
    a,b,r=fixture()
    if kind=='delta': b['delta']=.05
    elif kind=='count': b['independent_initials']=512
    elif kind=='candidate': b['candidates'].pop()
    elif kind=='bound': a['candidates'][0]['threshold_upper']=.001
    elif kind=='certification': a['candidates'][0]['certified']=True
    elif kind=='model': r['raw']['per_training_seed'].pop('1802')
    elif kind=='mixture': r['raw']['per_training_seed']['1802']['selected_profiles']={'baseline':20}
    elif kind=='denominator': r['raw']['per_training_seed']['1802']['actual_random_mode']['conditional_harm']=.05
    with pytest.raises(ValueError): audit.shared_table(a,b,r,'phase05')


def test_prospective_path_refuses_before_reading_unfinished_effects(monkeypatch):
    events=[]
    def incomplete(plan,stage):
        events.append(stage)
        raise ValueError('calibration not complete')
    module=SimpleNamespace(inputs=lambda:({},{}),complete_stage=incomplete)
    monkeypatch.setitem(sys.modules,'recover_prospective_v24',module)
    monkeypatch.setattr(audit,'read',lambda p:pytest.fail('Must not read partial outcomes'))
    with pytest.raises(ValueError,match='not complete'): audit.prospective_inputs()
    assert events==['calibration']
