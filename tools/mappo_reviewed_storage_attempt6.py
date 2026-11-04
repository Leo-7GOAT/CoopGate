"""Storage identity after native full replay, without redundant payload decoding.

Caller MUST first bind actual successful collector/reviewer exits, verification
SHA and collection-manifest SHA. That native reviewer already decompresses and
checks every payload and replays every controller/physical step. Unchanged whole
database hashes transfer that proof; this function additionally verifies the
complete original schedule, SQLite integrity and sealed row/trace hash catalog.
It neither replaces native replay nor uses effects to select observations.
"""
from pathlib import Path


def verify_reviewed_blocks(e, study, stage, shard, plan):
    study=Path(study)
    if plan['validation_only'] or stage not in ('calibration','confirmation'):
        raise ValueError('Original formal stage required')
    cases=e.stage_cases(study,stage,plan)
    assigned=e.blocks_for(cases,shard,4)
    directory=e.stage_path(study,stage,shard)
    complete=e.read(directory/'collection_complete.json')
    if len(complete['blocks'])!=len(assigned):
        raise ValueError('Missing or duplicate scheduled block')
    count=0
    for index,(block,planned) in enumerate(zip(complete['blocks'],assigned)):
        keys=[[c['cluster_id'],m] for c in planned for m in c['methods']]
        metadata=dict(plan_sha256=e.sha(study/'plan.json'),stage=stage,shard=shard,
                      index=index,cases_sha256=e.canonical_sha256(planned))
        if block['block']!=f'block{index:04d}' or block['keys']!=keys or block['metadata']!=metadata:
            raise ValueError('Block differs from original frozen complete schedule')
        path=directory/block['block']/'observations.sqlite'
        seal=Path(str(path)+'.seal.json')
        if e.sha(path)!=block['database_sha256'] or e.sha(seal)!=block['seal_sha256']:
            raise ValueError('Native-reviewed database or seal bytes changed')
        receipt=e.read(seal)
        if (receipt['database_sha256']!=block['database_sha256']
                or receipt['metadata']!=metadata or receipt['cases']!=keys):
            raise ValueError('Seal identity differs from reviewed source')
        state=e.inventory(path)  # Integrity check and hash catalog only, no unpack.
        if state!={k:receipt[k] for k in ('metadata','cases','row_sha256','trace_sha256')}:
            raise ValueError('Stored identity/hash catalog differs from its seal')
        count+=len(keys)
    if count!=complete['executions'] or count!=complete['expected']:
        raise ValueError('Whole original stage count differs')
    return dict(executions=count,blocks=len(assigned),all_reviewed_database_bytes_and_schedule_verified=True,
                requires_prior_actual_full_native_replay=True,payloads_redecoded=False,scientific_effects_read=False)
