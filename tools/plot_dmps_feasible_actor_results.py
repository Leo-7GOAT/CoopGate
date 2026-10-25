"""Plot complete, audited physical trajectories; no simulation or new inference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'artifacts/literature_reproduction/dmps_feasible_actor_training'
OUT = DATA.parent/'dmps_feasible_actor_results'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plot():
    proof = read(DATA/'verification.json')
    assert proof['observed_exit']['exit_code'] == 0
    assert proof['all_physics_and_replay_arrays_reconstructed']
    assert proof['all_recorded_policy_decisions_replayed']
    assert proof['complete_sha256'] == digest(DATA/'complete.json')
    if OUT.exists():
        raise FileExistsError('Preserve the first results visualization')
    figure, axes = plt.subplots(2, 3, figsize=(12.6, 6.8), sharex=True, sharey=True)
    colors = ['#1479b8', '#df6a21', '#25965d']
    bound = {}
    for row, index in enumerate((0, 1)):
        for col, role in enumerate(('new', 'old', 'baseline')):
            path = (DATA/f'baseline/{index:03d}/episode.json' if role == 'baseline'
                    else DATA/f'evaluation_{role}/episode_{index:03d}/episode.json')
            record = read(path)
            bound[path.relative_to(ROOT).as_posix()] = digest(path)
            initial = np.asarray(record['row']['initial'] if role == 'baseline' else record['initial'])
            trace = record['trace'] if role == 'baseline' else [d['trace'] for d in record['decisions']]
            times = [0.] + [entry['t'] for entry in trace]
            states = np.asarray([initial] + [entry['x'] for entry in trace])
            for vehicle in range(3):
                # This named display variable is zero after arrival; raw positions
                # and all original cost/arrival definitions remain unchanged.
                distance = np.maximum(60.-states[:, vehicle, 0], 0.)
                axes[row, col].plot(times, distance, color=colors[vehicle], lw=1.8,
                                    label=f'Vehicle {vehicle+1}')
            axes[row, col].set_xlim(0, 25)
            axes[row, col].set_ylim(bottom=0)
            axes[row, col].grid(alpha=.2)
            if row == 0:
                axes[row, col].set_title({'new': 'Feasible actor: timeout',
                    'old': 'Previous critic repair: timeout',
                    'baseline': 'Recovery baseline: completed'}[role], fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(f'Seed {199904001+index}\nDistance remaining (m)')
            if row == 1:
                axes[row, col].set_xlabel('Physical simulation time (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    figure.suptitle('Audited nominal / clean SUMO tests: one unfinished vehicle in both learned policies', fontsize=12)
    figure.tight_layout(rect=(0, .055, 1, .95))
    OUT.mkdir(parents=True, exist_ok=False)
    figure.savefig(OUT/'nominal_trajectories.png', dpi=180)
    plt.close(figure)
    receipt = dict(verification_sha256=digest(DATA/'verification.json'),
        complete_sha256=proof['complete_sha256'], source_sha256=digest(Path(__file__)),
        input_episodes_sha256=bound, figure_sha256=digest(OUT/'nominal_trajectories.png'),
        selected_condition_indices=[0, 1], independent_initials=2,
        displayed_quantity='max(60 - actual longitudinal position, 0)',
        scope='Nominal/clean trajectory illustration only; all ten condition failures remain in the full audit.',
        new_sumo_executions=0, policy_or_metrics_changed=False)
    with (OUT/'figure_receipt.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    plot()
