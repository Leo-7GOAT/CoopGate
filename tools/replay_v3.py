"""Run a published V3 policy in SUMO, optionally with GUI and screenshot."""
from pathlib import Path
import argparse, json
from coopgate.recovery import RecoveryConfig
from coopgate.platoon import PlatoonEnv
from coopgate.learning_v3 import load
from coopgate.chain_recovery import ChainSupervisor

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy-seed', type=int, choices=[83, 97, 109], default=83)
    p.add_argument('--scene-seed', type=int, default=65001)
    p.add_argument('--scenario', choices=['cruise', 'moderate', 'hard_brake', 'double_brake'], default='double_brake')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--chain', action='store_true')
    p.add_argument('--out', type=Path, default=ROOT / 'runs/v3_replay')
    p.add_argument('--screenshot', type=Path)
    a = p.parse_args()
    if a.screenshot and not a.gui:
        p.error('--screenshot requires --gui')
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'trace.json').exists():
        raise RuntimeError('Refusing to overwrite existing replay')
    protocol = json.loads((ROOT / 'artifacts/v3/final/protocol.json').read_text())
    cfg = RecoveryConfig(**protocol['config'])
    model, _ = load(ROOT / f'artifacts/v3/final/seed{a.policy_seed}/policy.pt')
    with PlatoonEnv(cfg, 'joint_rl', gui=a.gui) as env:
        obs = env.reset(a.scene_seed, a.scenario)
        if a.chain:
            env.supervisor = ChainSupervisor(cfg, 'joint_rl')
        if a.gui:
            for name, color in [('lead', (230, 65, 45)), ('ego0', (30, 160, 200)), ('ego1', (30, 190, 90)), ('ego2', (150, 80, 210))]:
                env.connection.vehicle.setColor(name, color)
            env.connection.gui.setBoundary('View #0', float(env.states[:, 0].min()) - 20, -20, float(env.states[:, 0].max()) + 30, 20)
        while True:
            if a.screenshot and len(env.trace) == 8:
                a.screenshot.parent.mkdir(parents=True, exist_ok=True)
                env.connection.gui.setBoundary('View #0', float(env.states[:, 0].min()) - 20, -20, float(env.states[:, 0].max()) + 30, 20)
                env.connection.gui.screenshot('View #0', str(a.screenshot.resolve()), width=1400, height=700)
            obs, _, terminal, truncated, metrics = env.step(model.act(obs, True)[0])
            if terminal or truncated:
                break
        (a.out / 'trace.json').write_text(json.dumps(env.trace, indent=2), newline='\n')
        (a.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), newline='\n')
    if a.screenshot:
        assert a.screenshot.is_file()
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
