"""Find a reproducible example of unsafe independent switch composition."""
from pathlib import Path
import json
import numpy as np
from coopgate.platoon import initial_state
from coopgate.recovery import RecoveryConfig,RecoverySupervisor,witnesses
c=RecoveryConfig();found=None
for seed in range(1000):
    x=initial_state(seed,c);previous=np.zeros(3);proposal=np.full(3,2.5)
    independent=RecoverySupervisor(c,'individual');joint=RecoverySupervisor(c,'joint_rl')
    naive,info=independent.select(x,previous,proposal)
    safe,jinfo=joint.select(x,previous,proposal)
    if info['in_recoverable_set'] and not info['chosen_certified'] and jinfo['chosen_certified']:
        pair,terminal=witnesses(x,previous,c,proposal[None,:],per_pair=True)
        found={'seed':seed,'states':x.tolist(),'previous':previous.tolist(),'proposal':proposal.tolist(),
               'full_proposal_pair_margins':pair[0].tolist(),'independent_commands':naive.tolist(),
               'independent_diagnostics':info,'joint_commands':safe.tolist(),'joint_diagnostics':jinfo,
               'interpretation':'A follower certificate assumes its controlled front keeps its proposal. Replacing that front by braking can invalidate the follower certificate.'}
        break
if found is None:raise RuntimeError('No counterexample found in fixed search budget')
out=Path('artifacts/v3/development');out.mkdir(parents=True,exist_ok=True)
(out/'switch_counterexample.json').write_text(json.dumps(found,indent=2),newline='\n')
print(json.dumps(found,indent=2))
