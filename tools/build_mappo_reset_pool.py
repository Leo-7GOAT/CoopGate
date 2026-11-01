"""Generate unlabeled safe near-conflict reset states from fresh rule rollouts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.marl_v12 import batch_step, project_joint
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate
from develop_v12 import initial


def build(out, count=8192):
    out=Path(out); out.mkdir(parents=True, exist_ok=False)
    plan=dict(purpose='unlabeled training reset states, NOT evaluation or imitation',
              count=count, seed_base=950000001,
              sampler='fresh alternating nominal/dense, cycling four original modes',
              controller='highest slew-admissible acceleration with original physical filter',
              selection='ramp position in [-65,-15], all actual prefixes retained in source generation',
              old_evaluation_or_training_states_used=False, action_labels_used=False)
    (out/'protocol.json').write_text(json.dumps(plan, indent=2)+'\n')
    states=[]; previous_commands=[]; clocks=[]; sources=[]
    episode=0; packed=parameters(MergeConfig())
    while len(states)<count:
        seed=950000001+episode
        scenario='nominal' if episode%2==0 else 'dense'
        mode=('clean','uniform','bias','pulse')[(episode//2)%4]
        tape=disturbance_tape(seed,mode,.5,250)
        state=initial(seed,scenario); previous=np.zeros(3)
        for step in range(50):
            if np.all(state[:,0]>=60.):
                break
            backup=np.maximum(previous-2.5,-5.)
            if not robust_certificate(state,previous,backup,packed,.5):
                raise RuntimeError('Rule prefix outside recoverable contract; no rejection sampling')
            if step>0 and -65.<=state[1,0]<=-15. and len(states)<count:
                states.append(state.copy()); previous_commands.append(previous.copy()); clocks.append(step)
                sources.append((seed,scenario,mode,step))
            command,_,ok=project_joint(state,previous,np.minimum(previous+2.5,2.5),packed,.5)
            if not ok:
                raise RuntimeError('No certified rule-prefix command')
            nxt,executed,valid=batch_step(state[None],previous[None],command[None],
                tape[None,step*5:step*5+5],packed,.5)
            if not valid[0] or not np.array_equal(executed[0],command):
                raise RuntimeError('Rule-prefix physical failure')
            state=nxt[0]; previous=command
        episode+=1
    np.savez_compressed(out/'states.npz',state=np.array(states),previous=np.array(previous_commands),step=np.array(clocks))
    (out/'sources.json').write_text(json.dumps(sources)+'\n')
    report=dict(states=count, rule_episodes=episode,
                pool_sha256=hashlib.sha256((out/'states.npz').read_bytes()).hexdigest(),
                trained_policy_or_action_labels=False,
                meaning='Training distribution construction only, not performance evidence')
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True)
    parser.add_argument('--count',type=int,default=8192)
    args=parser.parse_args(); build(args.out,args.count)
