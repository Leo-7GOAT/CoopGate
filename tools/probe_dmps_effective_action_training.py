"""128 real SUMO transitions exercise the fixed effective-input training path.

Development integration only; no checkpoint enters the fixed-budget experiment.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from coopgate.dmps_effective_action import EffectiveCriticInputs
import train_dmps_effective_action as run
import verify_dmps_curriculum as physical
from verify_dmps_effective_action import reconstruct_replay


def probe(source):
    base=run.nominal.base;io=base.io;output=run.PROBE
    for name in run.SOURCES:io.committed(run.ROOT/name)
    _,binding=base.frozen.bound_inputs()
    hashes={**binding['source_sha256'],**{name:io.digest(run.ROOT/name) for name in run.SOURCES}}
    module,metadata=base.author.load(source);identity=io.runtime_identity()
    torch.set_num_threads(1);torch.manual_seed(805200);np.random.seed(805200)
    cases=[dict(index=i,seed=189903001+i,plant='nominal',disturbance='clean') for i in range(129)]
    output.mkdir(parents=True,exist_ok=False);(output/'training').mkdir();(output/'unused_test').mkdir()
    io.exclusive_json(output/'protocol.json',dict(source_sha256=hashes,author_sources=metadata,runtime=identity,
        actual_training_decisions=128,training_cases=cases,warmup=32,batch=32,interval=8,grad_steps=4,
        optimizer_updates=44,actor_updates=22,checkpoint_reused_for_full_run=False))
    io.exclusive_json(output/'native_start.json',dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=io.utc()))
    class ProbeExecution(base.ObservableSumoExecution):
        def execute(self,action,context):
            result=super().execute(action,context)
            if self.raw.collision or self.raw.violations:
                raise RuntimeError('Probe physical failure; preserve prefix without replacement')
            return result
    train=ProbeExecution(cases,output/'training',805601)
    unused=base.ObservableSumoExecution(cases,output/'unused_test',805602)
    try:
        trainer=module.MCTSTrainer(train,unused,torch.device('cpu'),replay_buffer_size=512,
            num_iters=4,branch_factor=4,path_len=2,ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,
            warmup_steps=32,batch_size=32,train_interval=8,num_grad_steps=4,
            act_noise=run.nominal.GaussianExploration(805603),eval_freq=10**9,model_save_freq=10**9,model_save_dir=None)
        base.install_bridges(trainer,module.MCTSPlanner);hooks=EffectiveCriticInputs(trainer.model)
        trainer.run(128);train.finish_record()
        assert len(train.calls)==128 and trainer.num_updates==44
        assert hooks.calls['critic2']==hooks.calls['critic1_targ']==hooks.calls['critic2_targ']==44
        assert hooks.calls['critic1']>=66
        audit=physical.audit_execution_tree(output,'training',cases)
        assert len(audit['calls'])==128
        expected=reconstruct_replay(audit['calls'])
        for name,value in expected.items():np.testing.assert_array_equal(getattr(trainer.buf,name)[:trainer.buf.size],value)
        io.exclusive_json(output/'audit.json',dict(actual_training_decisions=128,physical_substeps=640,
            replay_rows=trainer.buf.size,critic_calls=hooks.calls,all_actual_physics_and_replay_verified=True,
            episodes=[e['metadata'] for e in audit['episodes']]))
    except Exception as error:
        if train.raw is not None and not train.recorded:
            train.failure=dict(type=type(error).__name__,message=str(error));train.raw.abort_reason=str(error);train.finish_record()
        io.exclusive_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),actual_decisions=len(train.calls)))
        raise
    finally:train.close();unused.close()
    assert io.runtime_identity()==identity and base.author.verify(source)==metadata
    for name,expected in hashes.items():assert io.digest(run.ROOT/name)==expected
    io.exclusive_json(output/'complete.json',dict(actual_training_decisions=128,
        all_actual_physics_and_replay_verified=True,checkpoint_reused_for_full_run=False,
        efficacy_claimed=False,new_risk_certificate=False,
        files_sha256={p.relative_to(output).as_posix():io.digest(p) for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(dict(actual_training_decisions=128,physics_verified=True,optimizer_updates=44)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();probe(args.source)
