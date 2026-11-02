"""Require all four reviewed shards before describing the fixed score grid."""
import hashlib
import json
from pathlib import Path
import sys

from evaluate_scenario_confidence import RULES,schedule
from coopgate.recovery_gate_audit_v35 import relative_outcome,cell_summary

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101/development'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize():
    lookup={};inputs={};commands=substeps=0;identity=None
    for shard in range(4):
        directory=BASE/f'scenario_score_grid_20261102_shard{shard}'
        plan=json.loads((directory/'plan.json').read_text());proof=json.loads((directory/'verification.json').read_text())
        receipt=json.loads((directory/'native_receipt.json').read_text());complete=json.loads((directory/'collection_complete.json').read_text())
        if (receipt['collection_actual_native_exit_code']!=0 or receipt['review_actual_native_exit_code']!=0
                or not proof['all_match'] or proof['actual_sumo_episodes']!=200
                or complete['episodes_sha256']!=sha(directory/'episodes.jsonl')
                or proof['report_sha256']!=sha(directory/'report.json')
                or plan['shard']!=shard or plan['shards']!=4):
            raise ValueError('Every disjoint shard requires complete native/replay proof')
        if identity is not None and identity!=plan['model']:raise ValueError('Fixed model family changed')
        identity=plan['model']
        for line in (directory/'episodes.jsonl').read_text().splitlines():
            row=json.loads(line);key=tuple(row['key'])
            if key in lookup:raise ValueError('Duplicate observation across shards')
            lookup[key]=row
        inputs[str(shard)]={f:sha(directory/f) for f in ('plan.json','verification.json','episodes.jsonl','native_receipt.json')}
        commands+=proof['commands_matched'];substeps+=proof['independent_physical_substeps']
    cases=schedule(1,96900001,'integration')
    expected={tuple(case)+(method,) for case in cases for method in RULES}
    if set(lookup)!=expected:raise ValueError('Complete800-case model/threshold/baseline grid required')
    summaries={}
    for method in RULES:
        pairs=[]
        for case in cases:
            own=lookup[tuple(case)+(method,)];base=lookup[tuple(case)+('baseline',)]['row']
            pairs.append(dict(**relative_outcome(own['row'],base),row=own['row'],baseline=base,calls=own['calls']))
        summaries[method]=cell_summary(pairs)
    result=dict(actual_sumo_executions=800,independent_initials=5,repeated_conditions=80,
        commands_matched=commands,physical_substeps_matched=substeps,all_shards_complete=True,
        models=identity,methods=summaries,inputs_sha256=inputs,risk_certified=False,
        interpretation='Development only. SAME five initials repeated across modes/plants/models/thresholds; no probability calibration, threshold certification or best-seed selection.')
    out=ROOT/'artifacts/mappo_ctde_20261101/scenario_score_development_matrix_20261102.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(executions=800,independent_initials=5,commands=commands,all_shards_complete=True)),flush=True)


if __name__=='__main__':summarize()
