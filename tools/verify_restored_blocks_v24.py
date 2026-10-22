"""Read-only full physical review of the three originally interrupted blocks.

No new simulation, effectiveness aggregation, model choice or risk calibration.
The selected blocks are fixed by the pre-existing interruption inventory.
"""
from collections import Counter
from pathlib import Path
import json

import torch

import recover_prospective_v24 as recovery
from coopgate.episode_store_v21 import inventory, readonly


def main():
    torch.set_num_threads(1)
    original = recovery.original
    script = Path(__file__).resolve()
    original.require_committed([script])
    rec, plan = recovery.inputs()
    spec = plan['stages']['calibration']
    entries = [entry for entry in rec['blocks'] if not entry['complete']]
    reviewer = original.IndependentReview(dict(spec=spec,models=plan['models']))
    counts, hashes = Counter(), {}
    for entry in entries:
        index, n = entry['initial_index'], entry['records']
        before = recovery.PRIOR/'calibration/blocks'/f'{index:05d}'
        after = recovery.OUTPUT/'calibration/blocks'/f'{index:05d}'
        original.verify_manifest(after)
        with readonly(before/'observations.sqlite') as connection:
            prior_blobs = connection.execute('SELECT * FROM episodes ORDER BY position').fetchall()
        with readonly(after/'observations.sqlite') as connection:
            new_blobs = connection.execute('SELECT * FROM episodes ORDER BY position LIMIT ?', (n,)).fetchall()
        if new_blobs != prior_blobs or len(new_blobs) != n:
            raise ValueError('Original compressed record bytes changed')
        receipt = original.read(after/'complete.json')
        if (receipt['full_initial_wall_ms'] is not None or receipt['retained_sumo_roles'] != n
                or receipt['new_sumo_roles'] != 50-n or receipt['resumed_segment_wall_ms'] <= 0):
            raise ValueError('Interrupted timing or role accounting is inconsistent')
        decision = original.read(after/'decision.json')
        if entry['decision_sha256'] is None:
            if decision['prediction']['full_prediction_wall_ms'] is not None:
                raise ValueError('Interrupted prediction total time cannot be reconstructed')
        elif original.digest(after/'decision.json') != entry['decision_sha256']:
            raise ValueError('Existing prediction receipt changed')
        counts.update(original.review_initial(recovery.OUTPUT,plan,'calibration',index,None,None,reviewer))
        hashes[str(index)] = original.digest(after/'sha256.json')
    expected = dict(main=40*len(entries),observed_shadow=10*len(entries),deployment=0,deployed_adapter_mode_checks=0)
    if dict(counts) != expected or reviewer.counts['physical_traces'] != expected['main']:
        raise ValueError('Independent review did not cover every restored role')
    # Recheck all original source/data fingerprints after reading restored data.
    recovery.inputs()
    destination = recovery.OUTPUT/'restored_block_review'
    destination.mkdir(exist_ok=False)
    original.exclusive_json(destination/'report.json',dict(utc=original.utc(),source_sha256=original.digest(script),
        recovery_plan_sha256=original.digest(recovery.RECOVERY/'plan.json'),
        initial_indices=[e['initial_index'] for e in entries],block_manifest_sha256=hashes,
        preserved_original_roles=sum(e['records'] for e in entries),
        newly_collected_missing_roles=sum(50-e['records'] for e in entries),
        original_record_blobs_exact=True,existing_predictions_exact=True,
        unknown_historical_total_times_preserved=True,role_counts=dict(counts),
        independent_review_counts=reviewer.counts,error_cases=reviewer.error_cases,
        additional_sumo_runs=0,additional_independent_initials=0,partial_effectiveness_analysis=False,
        full_calibration_complete=False,new_risk_certificate=False))
    original.exclusive_json(destination/'sha256.json',original.manifest(destination))
    print(json.dumps(dict(restored_initials=len(entries),roles=dict(counts),physical_traces=reviewer.counts['physical_traces'],
                         additional_sumo_runs=0,new_risk_certificate=False)),flush=True)


if __name__ == '__main__':
    main()
