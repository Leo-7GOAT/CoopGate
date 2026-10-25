"""Fixed eight-condition native curriculum initialization and physics probe."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.dmps_nominal_model_v2 import NominalModelEnvV2
import dmps_curriculum_execution as curriculum
import train_dmps_observable_development as base

ROOT=base.ROOT
SOURCES=('src/coopgate/dmps_curriculum_initials.py','tools/dmps_curriculum_execution.py',
    'tools/probe_dmps_curriculum_sumo.py','tests/test_dmps_curriculum_initials.py',
    'docs/dmps_curriculum_probe_protocol.md')


def cases():
    return [dict(index=2*p+s,seed=159900001+s,phase=p,plant='nominal',disturbance='clean')
        for p in range(4) for s in (0,1)]


def run(output):
    io=base.io
    for name in SOURCES:io.committed(ROOT/name)
    _,binding=base.frozen.bound_inputs()
    hashes={**binding['source_sha256'],**{name:io.digest(ROOT/name) for name in base.SOURCES+SOURCES}}
    identity=io.runtime_identity();torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=False);(output/'execution').mkdir()
    io.exclusive_json(output/'plan.json',dict(cases=cases(),source_sha256=hashes,runtime=identity,
        control='Maximum admissible braking, with common post-goal cruise',
        training_steps=0,comparative_efficacy_claimed=False,new_risk_certificate=False))
    env=curriculum.CurriculumExecution(cases(),output/'execution',805301)
    try:
        for case in cases():
            env.reset()
            for _ in range(50):
                x,previous,step=env.observables();model=NominalModelEnvV2(x,previous,step)
                action=np.full(3,-1.);model.step(action)
                context=dict(proposal=action.tolist(),selected_action=action.tolist(),intervention=0,
                    predicted_state=model.x.tolist(),nominal_model_decisions=1,wall_ms=0.,
                    prediction_contract='Initialization/physics probe, no learned policy or original planner')
                _,_,done,_=env.execute(action,context)
                if done:break
            env.finish_record()
            if case['phase']==0 and not env.episodes[-1]['review']['completed']:
                raise AssertionError('Near-goal braking corner must supply real team-completion data')
            print(json.dumps(dict(processed_conditions=env.cursor,expected_conditions=8)),flush=True)
    finally:env.close()
    assert len(env.episodes)==8
    for name,expected in hashes.items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Source changed during probe: '+name)
    assert io.runtime_identity()==identity
    io.exclusive_json(output/'complete.json',dict(episodes=env.episodes,actual_decisions=len(env.calls),
        plan_sha256=io.digest(output/'plan.json'),training_steps=0,new_risk_certificate=False,
        comparative_efficacy_claimed=False,
        files_sha256={p.relative_to(output).as_posix():io.digest(p) for p in output.rglob('*') if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
