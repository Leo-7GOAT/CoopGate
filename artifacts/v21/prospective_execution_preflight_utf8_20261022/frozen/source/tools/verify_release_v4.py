from pathlib import Path
import argparse,json,hashlib
import numpy as np
from coopgate.merge_v4 import MergeConfig,MergeEnv
from coopgate.learning_v4 import load
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v4'


def main(replay):
    hashes=json.loads((D/'SHA256SUMS.json').read_text())
    for name,digest in hashes.items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
    protocol=json.loads((D/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/'src/coopgate'/name).read_bytes()).hexdigest()==digest,name
    for seed in protocol['training_seeds']:
        folder=D/f'seed{seed}';manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['episodes']==320 and manifest['protocol_sha256']==hashes['protocol.json']
        assert manifest['checkpoint_sha256']==hashes[f'seed{seed}/policy.pt']
        if replay:
            model,_=load(folder/'policy.pt')
            for split,scenes in protocol['splits'].items():
                expected=json.loads((folder/f'trace_{split}_joint_rl.json').read_text())
                with MergeEnv(MergeConfig(**protocol['config'])) as e:
                    obs=e.reset(**scenes[0])
                    for row in expected:
                        obs,_,t,u,m=e.step(model.act(obs,True)[0])
                        assert np.allclose(e.x,row['x'],atol=1e-6,rtol=0)
                        assert np.allclose(e.prev,row['command'],atol=1e-7,rtol=0)
                    assert t or u
                print(json.dumps({'seed':seed,'split':split,'matched_steps':len(expected)}),flush=True)
    print(f'Verified {len(hashes)} V4 artifacts and frozen source hashes.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');main(p.parse_args().replay)
