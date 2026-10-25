"""All public prefixes of the COMPLETED 117-run probe; no new control/SUMO.

Post-hoc development only. The full 2,245-frame scope is fixed before output.
No outcomes of live calibration or OOD studies are read.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from coopgate.aged_observation_envelope import AgeAwareStateObserver
from coopgate.compound_backup_tube import certify_braking_tube
from coopgate.episode_store_v21 import observations

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'artifacts/diagnostics/timestamped_control_probe_v2'
OUT = ROOT/'artifacts/diagnostics/compound_backup_domain'
SOURCES = ('src/coopgate/compound_backup_tube.py', 'tools/audit_compound_backup_domain.py',
           'tests/test_compound_backup_tube.py', 'docs/compound_backup_tube.md',
           'src/coopgate/aged_observation_envelope.py', 'src/coopgate/episode_store_v21.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_case(receipts):
    """Calculation facade reads only whitelisted public receipt fields."""
    observer, history, rows = None, [], []
    for step, receipt in enumerate(receipts):
        assert receipt['decision'] == step
        if observer is None:
            observer = AgeAwareStateObserver(receipt['profile'])
        assert observer.profile == receipt['profile']
        previous = np.asarray(receipt['previous_command'])
        if step:
            history.append(previous.copy())
        estimate = observer.update(receipt['raw_state'], receipt['source_decision'], previous)
        assert estimate == receipt['envelope']
        command = np.maximum(previous-2.5, -5.)
        result = certify_braking_tube(estimate['lower'], estimate['upper'],
            estimate['memory_lower'], estimate['memory_upper'], history, command)
        rows.append(dict(decision=step, certified=result['certified'], reason=result['reason'],
                         checked_substeps=result['substeps'], first_command=command.tolist()))
    return rows


def audit():
    if OUT.exists():
        raise FileExistsError('Preserve the first complete development-domain analysis')
    for name in SOURCES:
        assert subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT) == (ROOT/name).read_bytes(), name
    source_hashes = {name:digest(ROOT/name) for name in SOURCES}
    complete, verification, exit_proof = (read(PARENT/name) for name in
        ('complete.json','verification.json','observed_execution_exit.json'))
    assert verification['observed_exit'] == exit_proof and exit_proof['exit_code'] == 0
    assert exit_proof['complete_sha256'] == verification['complete_sha256'] == digest(PARENT/'complete.json')
    assert verification['actual_sumo_runs'] == 117 and verification['received_frames'] == 2245
    assert verification['terminal_errors_replayed'] == 39
    parent_hashes = {name:digest(PARENT/name) for name in
        ('complete.json','verification.json','observed_execution_exit.json')}
    for name, expected in complete['files_sha256'].items():
        assert digest(PARENT/name) == expected
        parent_hashes[name] = expected
    cases=[]
    for job in read(PARENT/'protocol.json')['jobs']:
        folder = PARENT/'blocks'/f"{job['index']:03d}"
        done = read(folder/'complete.json')
        seal = read(folder/'observations.sqlite.seal.json')
        records = observations(folder/'observations.sqlite', seal['metadata'], seal['cases'])
        count = 0
        for key, body, trace in records:
            receipts = body['observation_adapter_inputs']
            terminal = body['row']['terminal_attempt'] is not None
            assert len(receipts) == len(trace)+int(terminal)
            rows = check_case(receipts)
            cases.append(dict(job=job, case=list(key), terminal_attempt=terminal,
                              frames=rows, initial_certified=rows[0]['certified'],
                              last_certified=rows[-1]['certified']))
            count += 1
        assert count == done['actual_sumo_runs'] == 13
    assert len(cases) == 117 and sum(len(c['frames']) for c in cases) == 2245
    assert sum(c['terminal_attempt'] for c in cases) == 39
    cells=[]
    for job in read(PARENT/'protocol.json')['jobs']:
        selected=[c for c in cases if c['job']==job]
        frames=[frame for c in selected for frame in c['frames']]
        cells.append(dict(**job, cases=13, received_frames=len(frames),
            certified_frames=sum(f['certified'] for f in frames),
            initial_certified_cases=sum(c['initial_certified'] for c in selected),
            abort_frames=sum(c['terminal_attempt'] for c in selected),
            abort_frames_certified=sum(c['last_certified'] for c in selected if c['terminal_attempt']),
            reasons=dict(Counter(f['reason'] for f in frames))))
    assert all(digest(ROOT/name)==expected for name,expected in source_hashes.items())
    assert all(digest(PARENT/name)==expected for name,expected in parent_hashes.items())
    result=dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_hashes,
        parent_sha256=parent_hashes, actual_new_sumo_runs=0, current_control_actions_executed=0,
        parent_cases=117, received_frames=2245, terminal_frames=39,
        new_independent_initials=0, partial_formal_efficacy_read=False,
        posthoc_development_only=True, closed_loop_recovery_proven=False,
        performance_risk_certified=False, numerical_interval_arithmetic_verified=False,
        aggregate=cells,cases=cases)
    OUT.mkdir()
    for name, value in [('report.json',result),('verification.json',dict(
        report_scope_complete=True,public_prefixes_recomputed=True,
        report_sha256=None,source_sha256=source_hashes))]:
        if name == 'verification.json':
            value['report_sha256']=digest(OUT/'report.json')
        with (OUT/name).open('x',encoding='utf-8') as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps(dict(report_sha256=digest(OUT/'report.json'),aggregate=cells),ensure_ascii=False))


if __name__ == '__main__':
    audit()
