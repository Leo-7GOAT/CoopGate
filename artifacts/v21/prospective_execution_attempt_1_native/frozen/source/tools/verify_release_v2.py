"""Verify published artifact bytes, then reproduce one saved scene per policy."""
from pathlib import Path
import argparse, csv, hashlib, json
from coopgate.config import Config
from coopgate.mappo import load_policy
from coopgate.sim import SumoEnv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/v2_final'


def main(replay):
    hashes = json.loads((DATA / 'SHA256SUMS.json').read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
    protocol = json.loads((DATA / 'protocol.json').read_text())
    for seed in protocol['training_seeds']:
        folder = DATA / f'seed{seed}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['protocol_sha256'] == hashes['protocol.json']
        assert manifest['checkpoint_sha256'] == hashes[f'seed{seed}/policy.pt']
        for name, digest in manifest['source_sha256'].items():
            assert hashlib.sha256((ROOT / name.replace('\\', '/')).read_bytes()).hexdigest() == digest, name
        if replay:
            model, _ = load_policy(folder / 'policy.pt')
            with SumoEnv(Config(**protocol['config']), 'paired') as env:
                scene = protocol['test_scenes'][0]
                obs = env.reset(**scene)
                while True:
                    action, _, _ = model.act(obs, True)
                    obs, _, terminal, truncated, metrics = env.step(action)
                    if terminal or truncated:
                        break
                with (folder / 'episodes.csv').open() as f:
                    expected = next(r for r in csv.DictReader(f) if r['split'] == 'test' and r['method'] == 'paired' and int(r['seed']) == scene['seed'])
                for key in ('collision', 'success'):
                    assert metrics[key] == (expected[key] == 'True'), (seed, key)
                for key in ('elapsed_s', 'rl_fraction', 'switches'):
                    assert abs(metrics[key] - float(expected[key])) < 1e-8, (seed, key)
                print(json.dumps({'seed': seed, 'reproduced': scene, 'success': metrics['success']}), flush=True)
    print(f'Verified {len(hashes)} artifact hashes and all frozen source manifests.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    main(parser.parse_args().replay)
