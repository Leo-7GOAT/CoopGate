import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_memoized_recovery_v37 import CAPACITY, RULES, cache_work, groups, non_timing


def cases():
    result = []
    for i in range(30):
        base = [i,'nominal',i,37000001+i,'lag_strong']
        result.append(base+['baseline','none'])
        result.extend(base+[rule,model] for rule in ('mps_shared','mamps_shared',*RULES)
                      for model in ('1802','1812','1822'))
    return result


def test_pair_every_performance_episode_model_and_rule_without_outcome_selection():
    source = cases()
    plan = groups(source)
    assert len(plan) == 180 and sum(len(g['executions']) for g in plan) == 360
    selected = {tuple(k) for k in source if k[-2] in RULES}
    assert {tuple(g['case']) for g in plan} == selected
    for index,g in enumerate(plan):
        expected = ['original','memoized'] if index%2 == 0 else ['memoized','original']
        assert [k[-1] for k in g['executions']] == expected
        assert all(k[:-1] == g['case'] for k in g['executions'])
    assert groups(source[::-1]) == plan


@pytest.mark.parametrize('change',('missing','duplicate','model','rule','condition'))
def test_changed_or_incomplete_performance_grid_is_rejected(change):
    source = [k for k in cases() if k[-2] in RULES]
    if change == 'missing': source.pop()
    if change == 'duplicate': source[-1] = source[0]
    if change == 'model': source[0][-1] = 'unknown'
    if change == 'rule': source[0][-2] = 'raw'
    if change == 'condition': source[0][0] = 1000
    with pytest.raises(ValueError): groups(source)


def test_only_timing_and_explicit_new_cache_diagnostics_are_excluded_from_identity():
    record = dict(row=dict(setup_ms=3.,controller_ms=[4.],team_cost=10.),calls=[dict(wall_ms=5.,error=None)])
    trace = [dict(gate_ms=4.,command=[0.,1.,2.],candidate_evidence=[1,2])]
    changed,other = copy.deepcopy((record,trace))
    changed['row']['setup_ms'] = 9.;changed['row']['controller_ms'] = [10.]
    changed['calls'][0]['wall_ms'] = 11.
    other[0].update(gate_ms=10.,proposal_cache={'hits':1},proposal_cache_entries=1,
                    proposal_cache_capacity=CAPACITY,planning_counter_scope='logical')
    assert non_timing(record,trace) == non_timing(changed,other)
    other[0]['candidate_evidence'][0] = 99
    assert non_timing(record,trace) != non_timing(changed,other)


def fixture():
    return [dict(proposal_cache=dict(requests=4,hits=2,misses=2,successful_misses=2,failed_misses=0,
        logical_successful_transitions=40,computed_successful_transitions=20,capacity_skips=0),
        completed_planning_model_transitions=40,proposal_cache_capacity=CAPACITY,proposal_cache_entries=2,
        planning_work_complete=True,planning_counter_scope='Original completed_planning_model_transitions retains logical reference work; separate actual counters.')]


def test_cache_work_distinguishes_reference_work_from_actual_recomputation():
    summary = cache_work(fixture())
    assert summary['logical_successful_transitions'] == 40
    assert summary['computed_successful_transitions'] == 20 and summary['hits'] == 2


@pytest.mark.parametrize('change',('requests','misses','logical','actual','entries','scope','negative'))
def test_invalid_cache_accounting_fails_review(change):
    trace = fixture()
    if change == 'requests': trace[0]['proposal_cache']['requests'] = 3
    if change == 'misses': trace[0]['proposal_cache']['successful_misses'] = 3
    if change == 'logical': trace[0]['completed_planning_model_transitions'] = 39
    if change == 'actual': trace[0]['proposal_cache']['computed_successful_transitions'] = 41
    if change == 'entries': trace[0]['proposal_cache_entries'] = CAPACITY+1
    if change == 'scope': trace[0]['planning_counter_scope'] = 'actual all transitions'
    if change == 'negative': trace[0]['proposal_cache']['capacity_skips'] = -1
    with pytest.raises(ValueError): cache_work(trace)
