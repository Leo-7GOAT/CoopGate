"""Six actual SUMO integrity runs before the bounded-candidate reuse study."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import torch
from coopgate.episode_store_v21 import EpisodeStore,observations
from experiments.top_two_candidates.gate import TopTwoHistoryGate,TopTwoReserveGate
from experiments.top_two_candidates.audit import audit_record
import experiments.joint_admissibility.run_development as old

OUT=old.e.BASE/'development/local_top_two_smoke_20261103'
VARIANTS=('top_history','top_reserve')


def factory(variant):
    cls=TopTwoHistoryGate if variant=='top_history' else TopTwoReserveGate
    def build(method,model,case):return cls(model,case['plant'],old.BANK)
    return build


def run():
    torch.set_num_threads(1);p=old.guard();models=old.e.load_models()
    case=next(c for c in p['cases'] if c['seed']==875001667)
    OUT.mkdir(exist_ok=False)
    sources={n:old.e.sha(ROOT/n) for n in (
        'experiments/top_two_candidates/gate.py','experiments/top_two_candidates/audit.py',
        'experiments/top_two_candidates/smoke.py','tests/test_local_top_two_candidates.py',
        'docs/top_two_candidates_development_20261103_protocol.md')}
    metadata=dict(case=case,sources=sources,integrity_only=True,new_independent_initials=0)
    old.e.write(OUT/'plan.json',metadata)
    keys=[[case['cluster_id'],v,s] for v in VARIANTS for s in ('4703','4704','4705')]
    store=EpisodeStore(OUT/'observations.sqlite',metadata)
    try:
        with old.runtime.MatchedShiftedMergeEnv(plant=case['plant']) as env:
            for _,variant,seed in keys:
                record=old.runtime.execute_episode(case,seed,models[seed],env,controller_factory=factory(variant))
                payload={k:v for k,v in record.items() if k!='trace'};payload['variant']=variant
                store.append([case['cluster_id'],variant,seed],payload,record['trace'])
        store.seal(keys)
    finally:store.close()
    for key,record,trace in observations(OUT/'observations.sqlite',metadata,keys):
        record['trace']=trace
        audit_record(case,record,models[record['method']],reference_factory=factory(record['variant']))
    old.e.write(OUT/'report.json',dict(completed=True,actual_sumo_executions=6,
        full_controller_and_physics_replay=True,all_three_fixed_actor_seeds=True,
        new_independent_initials=0,statistical_certificate=False))
    print(json.dumps(dict(completed=True,actual_sumo=6,full_controller_physics_replay=True)),flush=True)


if __name__=='__main__':run()
