"""One-decision prediction audit on development scenes, not evaluation tuning."""
from pathlib import Path
import json
import numpy as np
from coopgate.config import Config
from coopgate.controllers import backup_intents
from coopgate.paired import ClosedLoopPredictor
from coopgate.sim import SumoEnv

cfg=Config(commit_lane_changes=True,continuous_network=True,paired_horizon=1)
new_errors=[];old_errors=[]
with SumoEnv(cfg,'traditional') as e:
    p=ClosedLoopPredictor(cfg)
    for seed in range(30010,30015):
        e.reset(seed)
        for k in range(30):
            old=e.states.copy();intent=backup_intents(old,cfg)
            context={'previous_accel':e.previous_accel.copy(),'lane_targets':e.lane_targets.copy(),'lane_remaining':e.lane_remaining.copy()}
            f=p.rollout(old,intent,context,0)
            legacy=old.copy()
            for i in range(3):
                a=np.clip((intent[i].speed-old[i,2])/.8,-cfg.max_brake,cfg.max_accel)
                legacy[i,2]=np.clip(old[i,2]+a*.5,0,cfg.max_speed)
                legacy[i,0]+=.25*(old[i,2]+legacy[i,2])
            _,_,t,u,_=e.step([4,4,4])
            new_errors.append(f.next_states[:3,:3]-e.states[:3,:3])
            old_errors.append(legacy[:3,[0,2]]-e.states[:3,[0,2]])
            if t or u:break
result={'decisions':len(new_errors),'new_rmse_xyz_velocity':np.sqrt(np.mean(np.array(new_errors)**2,axis=(0,1))).tolist(),
        'legacy_rmse_x_velocity':np.sqrt(np.mean(np.array(old_errors)**2,axis=(0,1))).tolist(),
        'scope':'development nominal trajectories only; x,y,v columns for new model; no universal residual bound'}
out=Path('artifacts/v2_continuous/audit');out.mkdir(parents=True,exist_ok=True)
(out/'prediction_error.json').write_text(json.dumps(result,indent=2),newline='\n')
print(json.dumps(result))
