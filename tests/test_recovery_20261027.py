"""Fifth recovery tests use identities and synthetic bytes, never effects."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import recover_prospective_20261027 as recovery
import snapshot_interruption_20261027 as snapshot


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def terminal_fixture(cal):
    execution = cal/'calibration/execution'
    frozen = dict(stages=dict(calibration=dict(shards=3)))
    write_json(cal/'frozen/plan.json',frozen)
    write_json(cal/'recovery/plan.json',dict(recovery=True))
    plan_sha = recovery.original.digest(cal/'frozen/plan.json')
    recovery_sha = recovery.original.digest(cal/'recovery/plan.json')
    launchers,natives = [20,21,22],[30,31,32]
    identities = [dict(launcher_pid=launcher,native_pid=native,parent_pid=launcher)
                  for launcher,native in zip(launchers,natives)]
    state = dict(stage='calibration',phase='collect',exit_codes=[None]*3,
        plan_sha256=plan_sha,recovery_plan_sha256=recovery_sha,
        worker_pids=launchers,worker_native_pids=natives)
    write_json(execution/'state.json',state)
    write_json(execution/'collect_launch.json',dict(stage='calibration',supervisor_pid=40,
        worker_pids=launchers,worker_native_pids=natives,native_identities=identities,
        plan_sha256=plan_sha,recovery_plan_sha256=recovery_sha))
    write_json(execution/'watcher_launch.json',dict(stage='calibration',supervisor_pid=50,
        supervisor_native_pid=40,plan_sha256=plan_sha,recovery_plan_sha256=recovery_sha))
    pairs = [('supervisor',50,40),*((f'collect_{i}',launcher,native)
             for i,(launcher,native) in enumerate(zip(launchers,natives)))]
    for role,launcher,native in pairs:
        started = dict(native_pid=native,parent_pid=launcher)
        write_json(execution/f'{role}.native.json',started)
        write_json(execution/f'{role}.acknowledged.json',dict(launcher_pid=launcher,
            native_pid=native,parent_pid=launcher,
            native_start_sha256=recovery.original.digest(execution/f'{role}.native.json')))
    return execution,state


def test_v38_routes_are_new_and_parent_is_the_v29_recovery():
    assert snapshot.CAL == ROOT/'artifacts/recovery_20261026/calibration_recovery'
    assert snapshot.OUT == ROOT/'artifacts/recovery_20261027/calibration_interruption'
    assert recovery.PRIOR == snapshot.CAL
    assert recovery.OUTPUT == ROOT/'artifacts/recovery_20261027/calibration_recovery'
    assert recovery.PRIOR != recovery.OUTPUT
    assert recovery.SCRIPT == 'tools/recover_prospective_20261027.py'


def test_declared_pid_extraction_accepts_only_explicit_pid_fields():
    value = dict(worker_pids=[2,3,False,'4'],native_identity=dict(native_pid=5,parent_pid=6),
                 unrelated=7,nested=[dict(watcher_pid=8),dict(value=9)])
    assert snapshot._pids(value) == {2,3,5,6,8}


def test_conflicting_frozen_source_identity_is_rejected():
    assert snapshot.merge_source_hashes({'a':'1'},{'b':'2','a':'1'}) == {'a':'1','b':'2'}
    with pytest.raises(ValueError,match='disagree'):
        snapshot.merge_source_hashes({'a':'1'},{'a':'2'})


def test_live_known_handle_or_target_command_refuses_snapshot(monkeypatch):
    monkeypatch.setattr(snapshot,'declared_process_ids',lambda:{123})
    known = [dict(ProcessId=123,ParentProcessId=1,Name='python.exe',CommandLine='python unrelated.py')]
    with pytest.raises(ValueError,match='still live'):
        snapshot.assert_stopped(known,own_pid=10,parent_pid=11)
    command = [dict(ProcessId=124,ParentProcessId=1,Name='python.exe',
                    CommandLine='python tools/recover_prospective_20261026.py worker calibration collect 0')]
    with pytest.raises(ValueError,match='still live'):
        snapshot.assert_stopped(command,own_pid=10,parent_pid=11)
    unrelated = [dict(ProcessId=125,ParentProcessId=1,Name='python.exe',CommandLine='python other_study.py')]
    assert snapshot.assert_stopped(unrelated,own_pid=10,parent_pid=11)['target_execution_processes_absent']


def test_terminal_receipt_or_non_null_exit_cannot_be_called_unknown(tmp_path,monkeypatch):
    cal = tmp_path/'source'
    execution,state = terminal_fixture(cal)
    monkeypatch.setattr(snapshot,'CAL',cal)
    assert snapshot.terminal_state()['exit_codes'] == [None,None,None]
    (execution/'completed.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='Terminal receipt'):
        snapshot.terminal_state()
    (execution/'completed.json').unlink()
    state['exit_codes'] = [0,None,None]
    write_json(execution/'state.json',state)
    with pytest.raises(ValueError,match='unterminated'):
        snapshot.terminal_state()


@pytest.mark.parametrize('change',('plan','recovery','shards','worker','native','launch_identity','ack'))
def test_unterminated_state_is_bound_to_plans_three_workers_and_acknowledgments(tmp_path,monkeypatch,change):
    cal = tmp_path/'source'
    execution,state = terminal_fixture(cal)
    monkeypatch.setattr(snapshot,'CAL',cal)
    launch = json.loads((execution/'collect_launch.json').read_text(encoding='utf-8'))
    if change == 'plan': state['plan_sha256'] = 'different'
    elif change == 'recovery': state['recovery_plan_sha256'] = 'different'
    elif change == 'shards': state['exit_codes'].pop()
    elif change == 'worker': state['worker_pids'][0] = 99
    elif change == 'native': state['worker_native_pids'][0] = 99
    elif change == 'launch_identity': launch['native_identities'][0]['native_pid'] = 99
    elif change == 'ack':
        ack = json.loads((execution/'collect_0.acknowledged.json').read_text(encoding='utf-8'))
        ack['native_start_sha256'] = 'different'
        write_json(execution/'collect_0.acknowledged.json',ack)
    write_json(execution/'state.json',state)
    write_json(execution/'collect_launch.json',launch)
    with pytest.raises(ValueError):
        snapshot.terminal_state()


def test_archive_parts_preserve_every_source_byte_and_detect_changes(tmp_path,monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    files = []
    for index,size in enumerate((4,5,8,1)):
        path = source/str(index)
        path.write_bytes(bytes([index+1])*size)
        files.append(path)
    assert snapshot.parts_for(files,limit=9) == [files[:2],files[2:]]
    with pytest.raises(ValueError,match='Single file'):
        snapshot.parts_for(files,limit=7)
    target = tmp_path/'evidence/calibration'
    target.mkdir(parents=True)
    parts = []
    for index,group in enumerate(snapshot.parts_for(files,limit=9)):
        archive = target/f'part_{index}.zip'
        with zipfile.ZipFile(archive,'x') as handle:
            for path in group:
                handle.write(path,path.name)
        parts.append(dict(name=archive.name,sha256=recovery.original.digest(archive),
                          files=[path.name for path in group]))
    tracked = tmp_path/'tracked.py'
    tracked.write_bytes(b'frozen source')
    evidence = dict(source_directory='source',original_files=snapshot.files_at(source),archive_parts=parts,
        source_plan_sha256='plan',parent_recovery_plan_sha256='parent',source_exit_receipt_present=False,
        original_exit_status='unknown',partial_effectiveness_outcomes_read=False,
        source_sha256={'tracked.py':recovery.original.digest(tracked)})
    (target/'inventory.json').write_text(json.dumps(evidence),encoding='utf-8')
    monkeypatch.setattr(snapshot,'ROOT',tmp_path)
    monkeypatch.setattr(snapshot,'OUT',tmp_path/'evidence')
    monkeypatch.setattr(snapshot,'CAL',tmp_path/'cal')
    identities = {tmp_path/'cal/frozen/plan.json':'plan',
                  tmp_path/'cal/recovery/plan.json':'parent'}
    def digest(path):
        path = Path(path)
        return identities[path] if path in identities else __import__('hashlib').sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(snapshot.original,'digest',digest)
    assert snapshot.verify_snapshot(False) == evidence
    tracked.write_bytes(b'changed source')
    with pytest.raises(ValueError,match='snapshot source'):
        snapshot.verify_snapshot(False)
    tracked.write_bytes(b'frozen source')
    files[0].write_bytes(b'changed')
    with pytest.raises(ValueError,match='source files'):
        snapshot.verify_snapshot(False)


def valid_evidence(plan):
    return dict(interruption_generation=5,source_plan_sha256=recovery.original.digest(recovery.PRIOR/'frozen/plan.json'),
        parent_recovery_plan_sha256=recovery.original.digest(recovery.PRIOR/'recovery/plan.json'),
        original_exit_status='unknown',source_exit_receipt_present=False,partial_effectiveness_outcomes_read=False,
        initial_count=4096,confirmation_initial_count=1024,original_attempt=plan['attempt'],
        allocated_error=plan['allocated_error'],schedule_models_thresholds_unchanged=True,
        statistical_sample_size_unchanged=True,retained_roles=51,fully_completed_blocks=1,partial_blocks=1,
        blocks=[dict(initial_index=0,records=50,complete=True),dict(initial_index=3,records=1,complete=False)])


@pytest.mark.parametrize('change',('generation','receipt','outcomes','sample','attempt','budget','retained','partial'))
def test_evidence_cannot_change_statistical_or_prefix_identity(change):
    _,plan = recovery.original.load('prospective')
    evidence = valid_evidence(plan)
    if change == 'generation': evidence['interruption_generation'] = 2
    elif change == 'receipt': evidence['source_exit_receipt_present'] = True
    elif change == 'outcomes': evidence['partial_effectiveness_outcomes_read'] = True
    elif change == 'sample': evidence['initial_count'] = 4095
    elif change == 'attempt': evidence['original_attempt'] += 1
    elif change == 'budget': evidence['allocated_error'] /= 2
    elif change == 'retained': evidence['retained_roles'] += 1
    elif change == 'partial': evidence['partial_blocks'] = 0
    with pytest.raises(ValueError,match='statistical identity'):
        recovery.validate_evidence(evidence,plan)


def test_exact_v24_mechanics_are_rebound_without_replacement(monkeypatch):
    mechanics = recovery.mechanics
    names = ('boundary','copy_prefix','complete_partial','worker','supervise','complete_stage','analyze')
    retained = {name:getattr(mechanics,name) for name in names}
    old = {name:getattr(mechanics,name) for name in ('PRIOR','OUTPUT','RECOVERY','SCRIPT','inputs')}
    recovery.configure()
    try:
        assert all(getattr(mechanics,name) is value for name,value in retained.items())
        assert mechanics.PRIOR == recovery.PRIOR
        assert mechanics.OUTPUT == recovery.OUTPUT
        assert mechanics.RECOVERY == recovery.RECOVERY
        assert mechanics.SCRIPT == recovery.SCRIPT
        assert mechanics.inputs is recovery.inputs
        assert mechanics.phases('calibration') == ('boundary','collect','review')
        specs = recovery.original.stages('phase05')
        assert specs['calibration']['n_initials'] == 4096
        assert specs['confirmation']['n_initials'] == 1024
        assert specs['calibration']['total_sumo_runs'] == 204800
    finally:
        for name,value in old.items():
            monkeypatch.setattr(mechanics,name,value)


def test_valid_evidence_keeps_all_prefix_counts():
    _,plan = recovery.original.load('prospective')
    evidence = valid_evidence(plan)
    assert recovery.validate_evidence(deepcopy(evidence),plan) == evidence


def test_recovery_source_merge_rejects_any_hash_drift(monkeypatch):
    with pytest.raises(ValueError,match='disagree'):
        snapshot.merge_source_hashes({'tools/recover_prospective_v24.py':'frozen'},
                                     {'tools/recover_prospective_v24.py':'changed'})
