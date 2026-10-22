"""Verify frozen V3 artifacts and optionally replay every saved RL trace."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
from coopgate.recovery import RecoveryConfig
from coopgate.platoon import PlatoonEnv
from coopgate.learning_v3 import load
from coopgate.chain_recovery import ChainSupervisor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/v3'


def verify(replay=False):
    hashes = json.loads((DATA / 'SHA256SUMS.json').read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
    protocol = json.loads((DATA / 'final/protocol.json').read_text())
    total = 0
    for seed in protocol['training_seeds']:
        folder = DATA / f'final/seed{seed}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['protocol_sha256'] == hashes['final/protocol.json']
        assert manifest['checkpoint_sha256'] == hashes[f'final/seed{seed}/policy.pt']
        for name, digest in manifest['source_sha256'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        with (folder / 'episodes.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == manifest['episodes'] == (740 if seed == 83 else 640)
        total += len(rows)
        if not replay:
            continue
        model, _ = load(folder / 'policy.pt')
        for split, key in [('test', 'test_scenes'), ('shift', 'shift_scenes')]:
            scene = protocol[key][0]
            trace = json.loads((folder / f'trace_{split}_joint_rl.json').read_text())
            cfg = RecoveryConfig(**protocol['config'])
            with PlatoonEnv(cfg, 'joint_rl') as env:
                obs = env.reset(**scene)
                env.supervisor = ChainSupervisor(cfg, 'joint_rl')
                for expected in trace:
                    obs, _, terminal, truncated, metrics = env.step(model.act(obs, True)[0])
                    assert np.allclose(env.trace[-1]['commands'], expected['commands'], atol=1e-7, rtol=0)
                    assert np.allclose(env.trace[-1]['states'], expected['states'], atol=1e-6, rtol=0)
                assert terminal or truncated
                row = next(r for r in rows if r['split'] == split and r['method'] == 'joint_rl' and int(r['seed']) == scene['seed'])
                assert metrics['collision'] == (row['collision'] == 'True')
                for k in ('collision_penalized_progress_m', 'gap_violation_substeps', 'outside_set_steps', 'uncertified_action_steps'):
                    assert abs(metrics[k] - float(row[k])) < 1e-6, (seed, split, k)
            print(json.dumps({'seed': seed, 'split': split, 'chain_replay_matches_steps': len(trace)}), flush=True)
    assert total == 2020
    print(f'Verified {len(hashes)} artifact hashes, frozen source manifests, and {total} primary episodes.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--replay', action='store_true')
    verify(p.parse_args().replay)
