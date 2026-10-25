"""128-transition native integration; no checkpoint is inherited by the full run."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from coopgate.dmps_effective_action import EffectiveCriticInputs
from coopgate.dmps_feasible_actor import FeasibleActorOutputs
import train_dmps_feasible_actor as run
import verify_dmps_curriculum as physical
from verify_dmps_effective_action import reconstruct_replay


def probe(source):
    base=run.previous.nominal.base;io=base.io;output=run.PROBE
    parents=run.parent_evidence()
    for name in run.SOURCES:io.committed(run.ROOT/name)
    _,binding=base.frozen.bound_inputs()
    hashes={**binding['source_sha256'],**{name:io.digest(run.ROOT/name) for name in run.SOURCES}}
    module,metadata=base.author.load(source);runtime=io.runtime_identity()
    torch.set_num_threads(1);torch.manual_seed(805200);np.random.seed(805200)
    cases=[dict(index=i,seed=209905001+i,plant='nominal',disturbance='clean') for i in range(129)]
    output.mkdir(parents=True,exist_ok=False);(output/'training').mkdir();(output/'unused_test').mkdir()
    io.exclusive_json(output/'native_start.json',dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=io.utc()))
    io.exclusive_json(output/'protocol.json',dict(source_sha256=hashes,author_sources=metadata,runtime=runtime,parents=parents,
        actual_training_decisions=128,training_cases=cases,warmup=32,batch=32,interval=8,grad_steps=4,
        expected_critic_updates=44,expected_actor_updates=22,expected_actor_calls=dict(actor=118,actor_targ=44),
        actor_sampler_seed=805901,exploration_seed=805903,checkpoint_reused_for_full_run=False,efficacy_claimed=False))
    class Execution(base.ObservableSumoExecution):
        def execute(self,action,context):
            result=super().execute(action,context)
            if self.raw.collision or self.raw.violations:raise RuntimeError('Probe physical failure; preserve original prefix')
            return result
    train=Execution(cases,output/'training',805901);unused=base.ObservableSumoExecution(cases,output/'unused_test',805902)
    try:
        trainer=module.MCTSTrainer(train,unused,torch.device('cpu'),replay_buffer_size=512,num_iters=4,branch_factor=4,path_len=2,
            ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,warmup_steps=32,batch_size=32,train_interval=8,num_grad_steps=4,
            act_noise=run.previous.nominal.GaussianExploration(805903),eval_freq=10**9,model_save_freq=10**9,model_save_dir=None)
        base.install_bridges(trainer,module.MCTSPlanner)
        critics=EffectiveCriticInputs(trainer.model);actors=FeasibleActorOutputs(trainer.model)
        trainer.run(128);train.finish_record()
        assert len(train.calls)==128 and trainer.num_updates==44
        assert actors.calls==dict(actor=118,actor_targ=44)
        assert critics.calls['critic2']==critics.calls['critic1_targ']==critics.calls['critic2_targ']==44
        audited=physical.audit_execution_tree(output,'training',cases)
        expected=reconstruct_replay(audited['calls'])
        for name,value in expected.items():np.testing.assert_array_equal(getattr(trainer.buf,name)[:trainer.buf.size],value)
        io.exclusive_json(output/'audit.json',dict(actual_training_decisions=128,physical_substeps=640,
            replay_rows=trainer.buf.size,actor_calls=actors.calls,critic_calls=critics.calls,
            episodes=[e['metadata'] for e in audited['episodes']],all_actual_physics_and_replay_verified=True))
    except Exception as error:
        if train.raw is not None and not train.recorded:
            train.failure=dict(type=type(error).__name__,message=str(error));train.raw.abort_reason=str(error);train.finish_record()
        io.exclusive_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),actual_decisions=len(train.calls)))
        raise
    finally:train.close();unused.close()
    assert io.runtime_identity()==runtime and base.author.verify(source)==metadata and run.parent_evidence()==parents
    for name,expected in hashes.items():assert io.digest(run.ROOT/name)==expected
    io.exclusive_json(output/'complete.json',dict(actual_training_decisions=128,all_actual_physics_and_replay_verified=True,
        actor_calls=dict(actors.calls),checkpoint_reused_for_full_run=False,efficacy_claimed=False,
        files_sha256={p.relative_to(output).as_posix():io.digest(p) for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(dict(actual_training_decisions=128,physical_substeps=640,actor_calls=actors.calls)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path);args=parser.parse_args();probe(args.source)
