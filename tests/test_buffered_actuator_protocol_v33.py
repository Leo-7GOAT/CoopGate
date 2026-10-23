from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_buffered_actuator_v33 import audit_groups,times


def fixtures():
    source=[[i,'nominal',i,63400001+i,plant,'phase05_'+model] for i in range(2)
        for plant in ('nominal','lag_strong') for model in ('1802','1812','1822')]
    integration=[k+[kind] for k in source if k[2]==0 for kind in ('nominal','reference')]
    return source,integration,[(1,'nominal',1,63400002,'lag_strong')]


def test_pairing_covers_every_preflight_and_all_failure_models_and_variants():
    source,integration,failures=fixtures();groups=audit_groups(integration,failures,source)
    assert len(groups)==18 and sum(len(g['executions']) for g in groups)==36
    assert sum(g['source']=='preflight' for g in groups)==12
    assert sum(g['source']=='all_old_baseline_failures' for g in groups)==6
    assert sum(g['executions'][0][-1]=='original' for g in groups)==9
    assert sum(g['executions'][0][-1]=='buffered' for g in groups)==9
    assert {tuple(g['case']) for g in groups if g['source']=='preflight'}=={tuple(k) for k in integration}


def test_failure_enrichment_does_not_duplicate_an_existing_preflight_case():
    source,integration,failures=fixtures()
    groups=audit_groups(integration,failures+[(0,'nominal',0,63400001,'lag_strong')],source)
    assert len(groups)==18


def test_all_original_failed_baselines_need_real_archived_conditions():
    source,integration,failures=fixtures()
    failures.append((99,'nominal',99,63400100,'lag_strong'))
    with pytest.raises(ValueError,match='outside complete'):
        audit_groups(integration,failures,source)


@pytest.mark.parametrize('bad',['duplicate','unavailable','wrong_kind'])
def test_cannot_replace_or_duplicate_old_integration_conditions(bad):
    source,integration,failures=fixtures()
    if bad=='duplicate':integration.append(integration[0])
    elif bad=='unavailable':integration[0][3]=999
    else:integration[0][-1]='invented'
    with pytest.raises(ValueError,match='Unique original'):
        audit_groups(integration,failures,source)


def test_terminal_exception_duration_and_exact_500ms_boundary_are_reported():
    result=times([dict(wall_ms=500.,error=None),dict(wall_ms=900.,error={'type':'RuntimeError'})])
    assert result['calls']==2 and result['sum_ms']==1400.
    assert result['over_500ms']==1 and result['terminal_calls']==1
    assert result['p50_p95_p99_max_ms'][-1]==900.
