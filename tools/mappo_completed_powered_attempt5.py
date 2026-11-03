"""Read attempt5 effects only after exact complete owned-native evidence.

Used by postprocessing/publication,not by training,online control or inference.
No success decision is assumed: a complete negative study must also be kept.
"""
from pathlib import Path
import mappo_powered_history_formal_experiment as e


def complete_evidence(study=None):
    study=Path(study) if study is not None else e.FORMAL
    owner=e.read(study/'orchestrator_native_receipt.json')
    # Require the root terminal witness before touching any treatment effect.
    if owner['actual_native_exit_code']!=0:raise ValueError('Root native owner did not exit zero')
    if owner['completion_receipt_sha256']!=e.sha(study/'completion_receipt.json'):
        raise ValueError('Owner witness does not bind this exact completed study')
    receipt=e.read(study/'completion_receipt.json')
    if (receipt['actual_native_finalizer']['actual_native_exit_code']!=0
        or receipt['total_actual_sumo_runs']!=45056 or receipt['main_actual_sumo_runs']!=20480
        or receipt['independent_initials']!=5120):raise ValueError('Full original fixed scope not complete')
    expected={(stage,action,shard) for stage in ('calibration','confirmation') for action in ('collect','review') for shard in range(4)}
    if (len(receipt['stages'])!=16
        or {(r['stage'],r['action'],r['shard']) for r in receipt['stages']}!=expected
        or any(r['actual_native_exit_code']!=0 for r in receipt['stages'])):
        raise ValueError('Every exact native collection/review must complete')
    plan=e.bound(study,False)
    snapshot=e.read(study/'source_snapshot.json')
    if (snapshot['source_commit']!=plan['source_commit'] or snapshot['plan_sha256']!=e.sha(study/'plan.json')
        or len(snapshot['files'])!=len(plan['source_sha256'])+2):raise ValueError('Frozen source snapshot identity differs')
    expected_hashes=dict(plan['source_sha256'])
    expected_hashes['artifacts/mappo_ctde_20261101/observed_history_fixed_candidate_20261103.json']=plan['candidate_sha256']
    expected_hashes['artifacts/mappo_ctde_20261101/development/compact_coupled_bank_20261102.json']=e.design.validate_candidate()['coupled_bank_sha256']
    paths=set()
    for item in snapshot['files']:
        if item['path'] in paths:raise ValueError('Duplicate snapshot member')
        if expected_hashes.get(item['path'])!=item['sha256']:raise ValueError('Snapshot member differs from frozen plan/candidate')
        paths.add(item['path']);path=(study/'source_snapshot'/item['path']).resolve()
        if not path.is_relative_to((study/'source_snapshot').resolve()) or e.sha(path)!=item['sha256']:
            raise ValueError('Frozen source snapshot bytes changed')
    if set(expected_hashes)!=paths:raise ValueError('Source snapshot incomplete')
    for name in ('certificate','comparisons'):
        if e.sha(study/f'analysis/{name}.json')!=receipt[name+'_sha256']:raise ValueError('Final analysis changed')
    cert=e.read(study/'analysis/certificate.json');comp=e.read(study/'analysis/comparisons.json')
    if (cert['candidate_sha256']!=plan['candidate_sha256'] or cert['attempt']!=5 or cert['allocated_error']!=.05/30 or cert['fixed_models']!=[4703,4704,4705]
        or cert['common_gain_floor']!=.0025 or cert['calibration_initials']!=4096 or cert['confirmation_initials']!=1024
        or cert['primary_risk']['risk_limit']!=.05 or cert['adoption']['threshold']!=.2
        or cert['paired_gain']['threshold_percent']!=.5):raise ValueError('Original scientific family or thresholds changed')
    if set(comp['confirmation_methods'])!=set(e.runtime.CONFIRMATION_METHODS):raise ValueError('Full comparison family missing')
    return cert,comp
