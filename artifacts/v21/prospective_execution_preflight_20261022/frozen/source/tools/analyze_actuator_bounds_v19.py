"""Analytic error-envelope check for the exact declared V19 actuator plants."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from probe_disturbed_gate_v19 import digest, save


def lag_bound(tau, dt=.1, substeps=5, command_increment=2.5, innovation=.1):
    if tau == 0:
        return innovation
    rho = np.exp(-dt / tau)
    return float(rho * command_increment / (1 - rho ** substeps) + innovation)


def analyze(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    threshold = brentq(lambda tau: lag_bound(tau) - .5, .01, .2, xtol=1e-14)
    report = dict(
        purpose='Analytic coverage of the existing physical acceleration-error assumption; no performance-risk certificate',
        assumptions=dict(substep_seconds=.1, decision_substeps=5, maximum_command_increment=2.5,
                         command_range=[-5., 2.5], velocity_range=[0., 20.], innovation_radius=.1,
                         initial_actuator=0., initial_previous_command=0., certified_error_radius=.5),
        gain_drag=dict(maximum_gain_error=.04, maximum_drag=.0005,
                       gain_term=.04 * 5., drag_term=.0005 * 20. ** 2, innovation_term=.1,
                       acceleration_error_upper_bound=.04 * 5. + .0005 * 20. ** 2 + .1,
                       satisfies_original_error_assumption=True),
        lag_mild=dict(tau_range=[.03, .05], acceleration_error_upper_bound=lag_bound(.05),
                      satisfies_original_error_assumption=lag_bound(.05) <= .5),
        lag_strong=dict(tau_range=[.3, .5], acceleration_error_upper_bound=lag_bound(.5),
                        satisfies_original_error_assumption=lag_bound(.5) <= .5),
        maximum_uniformly_covered_lag_tau_seconds=threshold,
        scope='Gain/drag and lag are separate plant families. Bound applies to any admissible command sequence with the specified initialization; beyond the bound it does not assert every trajectory fails. Other physical-certificate assumptions still apply. No transfer of performance-risk guarantee.',
        source_sha256={str(Path(__file__).name): digest(__file__),
                       'shifted_sumo_v19.py': digest(Path(__file__).resolve().parents[1] / 'src/coopgate/shifted_sumo_v19.py')})
    save(output / 'analysis.json', report)
    shutil.copy2(__file__, output / Path(__file__).name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    taus = np.linspace(0., .5, 501)
    fig, ax = plt.subplots(figsize=(7.8, 4.6), layout='constrained')
    ax.plot(taus, [lag_bound(t) for t in taus], color='#214e64', lw=2,
            label='Worst-case bound, including innovation')
    ax.axhline(.5, color='#a54343', ls='--', label='Original error assumption: 0.5 m/s²')
    ax.axvspan(.03, .05, color='#689767', alpha=.3, label='Declared mild lag')
    ax.axvspan(.3, .5, color='#cb8142', alpha=.18, label='Declared strong lag')
    ax.scatter([threshold], [.5], color='#a54343', zorder=4)
    ax.annotate(f'Coverage boundary: {1000 * threshold:.1f} ms', (threshold, .5),
                xytext=(.16, .75), arrowprops=dict(arrowstyle='->', color='#555'), fontsize=9)
    ax.set(xlabel='Actuator time constant τ (s)', ylabel='Acceleration-error upper bound (m/s²)',
           xlim=(0., .5), ylim=(0., 3.6), title='Physical assumption boundary for the declared actuator model')
    ax.grid(alpha=.15)
    ax.legend(loc='upper left', fontsize=8)
    fig.savefig(output / 'actuator_envelope.png', dpi=180)
    plt.close(fig)
    save(output / 'sha256.json', {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    analyze(args.output)
