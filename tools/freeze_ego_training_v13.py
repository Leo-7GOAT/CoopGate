"""Declare three fresh training seeds before any release training."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    out=ROOT/'runs/v13_ego_release';out.mkdir(parents=True,exist_ok=False)
    files=['src/coopgate/ego_v13.py','src/coopgate/ego_models_v13.py','src/coopgate/marl_v12.py',
        'src/coopgate/training_v12.py','src/coopgate/fast_v8.py','src/coopgate/robust_v9.py',
        'tools/train_dqn_ego_v13.py','tools/freeze_ego_training_v13.py']
    protocol={'training_seeds':[701,711,721],'steps_per_seed':2097152,'checkpoint':'final only',
        'algorithm':'Double DQN; no demonstrations or imitation warm start','gamma':.995,
        'own_objective':'ego arrival + 0.05 total command variation until ego arrival',
        'train_domain':'nominal/dense with equal probability; iid uniform acceleration error +/-0.5',
        'planned_use':'standalone actor and learned-tail rollout; report these separately',
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (out/'training_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(protocol))
