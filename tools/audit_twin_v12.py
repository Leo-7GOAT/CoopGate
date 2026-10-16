"""Check the model simulator against actual SUMO at every physical substep."""
import argparse,json
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.marl_v12 import MARLGate,load_policy,batch_step,observations,action_mask,LEVELS
from coopgate.fast_v8 import parameters
from develop_v12 import initial


def run(path=None):
    model=None if path is None else load_policy(path)[0]
    single=False if model is None else load_policy(path)[1]['algorithm']=='single_ego_ppo_v12'
    rows=[]
    with DisturbedMergeEnv() as env:
        for regime in ('nominal','dense'):
            for mode in ('clean','uniform','bias','pulse'):
                seed=18010001+len(rows)
                env.reset_disturbed(seed,regime,mode,.5);env.gate=MARLGate(model,mode='constant' if model is None else ('single' if single else 'learned'))
                reset_error=float(np.max(np.abs(env.x-initial(seed,regime))))
                max_error=reset_error;max_command=0.;steps=0
                while True:
                    x,prev=env.x.copy(),env.prev.copy();step=env.gate.step
                    mask=action_mask(prev)
                    a=np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
                    if model is not None:
                        ob=observations(x,prev,step)
                        if single:a[1]=model.proposal(ob[1:2],mask[1:2])[0]
                        else:a=model.proposal(ob,mask)
                    next_x,commands,valid=batch_step(x[None],prev[None],LEVELS[a][None],env.tape[env.substep:env.substep+5][None],parameters(env.c),.5)
                    assert valid.all()
                    _,_,terminal,truncated,_=env.step([4]*3);steps+=1
                    max_error=max(max_error,float(np.max(np.abs(next_x[0]-env.x))))
                    max_command=max(max_command,float(np.max(np.abs(commands[0]-env.prev))))
                    if terminal or truncated:break
                assert max_error<1e-9 and max_command<1e-12
                rows.append({'seed':seed,'regime':regime,'mode':mode,'steps':steps,'max_state_error':max_error,
                    'max_command_error':max_command,'reset_error':reset_error,'metrics':env.metrics()})
    return {'checkpoint':str(path),'executions':len(rows),'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy');p.add_argument('--out',default='runs/v12_twin_audit.json');a=p.parse_args()
    result=run(a.policy);Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'executions':len(result['rows']),
        'max_state_error':max(r['max_state_error'] for r in result['rows'])}),flush=True)
