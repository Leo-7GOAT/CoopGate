"""Recovery scheduling and timing preservation, including a split implementation pair."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import recover_buffered_v29 as recovery
from coopgate.episode_store_v21 import EpisodeStore, observations, readonly


def synthetic(tmp_path, monkeypatch, count):
    out, prior = tmp_path/'out', tmp_path/'prior'
    out.mkdir()
    (out/'plan.json').write_text('{}', encoding='utf-8')
    (out/'recovery').mkdir()
    (out/'recovery/plan.json').write_text('{}', encoding='utf-8')
    (out/'blocks').mkdir()
    monkeypatch.setattr(recovery, 'OUT', out)
    monkeypatch.setattr(recovery, 'PRIOR', prior)
    monkeypatch.setattr(recovery, 'RECOVERY', out/'recovery')
    monkeypatch.setattr(recovery.original, 'OUT', out)
    group = dict(block_index=0, study='phase', initial_index=0,
        cases=[dict(case=[0, mode, 'phase05_'+model]) for mode in ('bias','clean','pulse','uniform')
               for model in ('1802','1812','1822')])
    schedule = recovery.original.schedule(group)
    def record(key, wall):
        return dict(row=dict(identity=list(key[:-1]), aborted=False, completed=True),
                    calls=[dict(error=None, wall_ms=wall)], episode_wall_seconds=wall/1000), [dict(identity=list(key[:-1]))]
    checks, executed = [], []
    def execute(group, item, implementation, *args):
        key = (*item['case'], implementation)
        executed.append(key)
        return record(key, 999.)
    def check(group, item, row, trace, *args):
        assert row['row']['identity'] == item['case']
        assert trace == [dict(identity=item['case'])]
        checks.append(item['case'])
    def pair(case, current):
        a, b = current['reference'], current['buffered']
        assert a[0]['row'] == b[0]['row'] and a[1] == b[1]
        return dict(case=list(case), both_physical_reconstructions_pass=True, all_non_timing_fields_exact=True)
    monkeypatch.setattr(recovery, 'execute_record', execute)
    monkeypatch.setattr(recovery, 'check_record', check)
    monkeypatch.setattr(recovery, 'check_pair', pair)
    entry, raw, before = None, [], None
    if count:
        source = prior/'blocks/00000'
        source.mkdir(parents=True)
        store = EpisodeStore(source/'observations.sqlite', recovery.original.metadata(group))
        for i, key in enumerate(schedule[:count]):
            row, trace = record(key, i+.125)
            store.append(key, row, trace)
        if count == len(schedule):
            store.seal(schedule)
            recovery.original.exclusive_json(source/'complete.json', dict(paired_cases=[
                dict(case=item['case'], both_physical_reconstructions_pass=True, all_non_timing_fields_exact=True)
                for item in group['cases']]))
        else:
            store.close()
        with readonly(source/'observations.sqlite') as connection:
            raw = connection.execute('SELECT * FROM episodes ORDER BY position').fetchall()
        entry = dict(initial_index=0, records=count, complete=count==len(schedule),
                     database_sha256=recovery.original.digest(source/'observations.sqlite'))
        before = recovery.snapshot.files_at(source)
    return dict(out=out, prior=prior, group=group, schedule=schedule, entry=entry,
                raw=raw, before=before, checks=checks, executed=executed)


@pytest.mark.parametrize('count', [0, 1, 11, 23, 24])
def test_exact_missing_roles_and_all_old_times_survive_including_split_pairs(tmp_path, monkeypatch, count):
    f = synthetic(tmp_path, monkeypatch, count)
    result = recovery.complete_group(f['group'], f['entry'], {}, {}, {}, {})
    assert f['executed'] == f['schedule'][count:]
    assert result['retained_roles'] == count and result['new_roles'] == 24-count
    assert result['reviewed_roles'] == 24 and result['paired_cases'] == 12
    assert len(f['checks']) == 24
    new = f['out']/'blocks/00000/observations.sqlite'
    with readonly(new) as connection:
        after = connection.execute('SELECT * FROM episodes ORDER BY position').fetchall()
    assert after[:count] == f['raw']
    if count:
        assert recovery.snapshot.files_at(f['prior']/'blocks/00000') == f['before']
    records = list(observations(new, recovery.original.metadata(f['group']), f['schedule']))
    assert [row['calls'][0]['wall_ms'] for _,row,_ in records[:count]] == [i+.125 for i in range(count)]
    assert all(row['calls'][0]['wall_ms'] == 999. for _,row,_ in records[count:])


def test_recovery_does_not_silently_fill_a_missing_interior_role(tmp_path, monkeypatch):
    f = synthetic(tmp_path, monkeypatch, 11)
    f['entry']['database_sha256'] = 'changed'
    with pytest.raises(ValueError, match='partial database changed'):
        recovery.complete_group(f['group'], f['entry'], {}, {}, {}, {})
    assert f['executed'] == []


def test_no_native_worker_or_statistical_controller_is_replaced_by_configuration(monkeypatch):
    retained = {name:getattr(recovery.original, name) for name in ('TimedGate','PhaseRepairGate','BufferedPhaseGate','check_calls','schedule','watch','summarize')}
    for name in ('OUT','SCRIPT','inputs'):
        monkeypatch.setattr(recovery.original, name, getattr(recovery.original, name))
    recovery.configure()
    assert all(getattr(recovery.original, name) is value for name,value in retained.items())
    assert recovery.original.OUT == recovery.OUT and recovery.OUT != recovery.PRIOR
