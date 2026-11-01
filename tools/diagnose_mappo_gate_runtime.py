"""Post-hoc descriptive attribution of measured gate cost; no new trials."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(directory):
    directory=Path(directory)
    proof=json.loads((directory/'verification.json').read_text())
    receipt=json.loads((directory/'native_receipt.json').read_text())
    if not proof['all_match'] or receipt['review_actual_native_exit_code']!=0:
        raise ValueError('Completed actual collection and independent review required')
    data=(directory/'episodes.jsonl').read_bytes()
    complete=json.loads((directory/'collection_complete.json').read_text())
    if hashlib.sha256(data).hexdigest()!=complete['episodes_sha256']:
        raise ValueError('Outcomes changed')
    records=[json.loads(s) for s in data.splitlines()]
    cells={};slow=[]
    for record in records:
        seed,scene,profile,mode,rule=record['key']
        key=f'{rule}/{profile}/{scene}'
        cell=cells.setdefault(key,dict(times=[],adopted_steps=0,forecast_calls=0,
                                      computed_planning_transitions=0,logical_planning_transitions=0))
        if len(record['calls'])!=len(record['trace']):
            raise ValueError('This attribution requires complete successful callbacks; retain terminal cases separately')
        for step,(call,item) in enumerate(zip(record['calls'],record['trace'])):
            cell['times'].append(call['wall_ms']);cell['adopted_steps']+=int(item['actual_learning_used'])
            cell['forecast_calls']+=item['forecast_calls']
            cache=item['proposal_cache']
            cell['computed_planning_transitions']+=cache['computed_successful_transitions']
            cell['logical_planning_transitions']+=item['completed_planning_model_transitions']
            if call['wall_ms']>500:
                slow.append(dict(seed=seed,scene=scene,profile=profile,mode=mode,rule=rule,step=step,
                    wall_ms=call['wall_ms'],used_learning=item['actual_learning_used'],
                    candidate_count=item['candidate_count'],forecast_calls=item['forecast_calls'],
                    computed_planning_transitions=cache['computed_successful_transitions'],
                    logical_planning_transitions=item['completed_planning_model_transitions']))
    for cell in cells.values():
        times=np.asarray(cell.pop('times'))
        cell.update(callback_count=len(times),sum_ms=float(times.sum()),p95_ms=float(np.quantile(times,.95)),
                    max_ms=float(times.max()),over_500ms=int((times>500).sum()))
    result=dict(purpose='Post-hoc engineering diagnosis, no new efficacy/risk/runtime-isolation claim',
        source_episodes_sha256=complete['episodes_sha256'],extra_simulator_runs=0,
        extra_independent_initials=0,cells=cells,slow_callbacks=sorted(slow,key=lambda r:-r['wall_ms']),
        interpretation='Correlation with work counters is not causal attribution; slow-callback selection is explicitly post-hoc')
    (directory/'runtime_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cells=len(cells),over_500ms=len(slow),
                         slowest=sorted(slow,key=lambda r:-r['wall_ms'])[:3])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory')
    analyze(parser.parse_args().directory)
