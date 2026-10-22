"""Actual SUMO parity and actuator-reconstruction checks, not performance claims."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from coopgate.advantage_gate_v18 import JointAdvantageGate
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.shifted_sumo_v19 import PLANTS, ShiftedMergeEnv, audit_shifted_trace
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]


def run(directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model_path = ROOT / 'artifacts/v18/independent_models/coupled_1802/policy.pt'
    model, _ = load_coupled(model_path)
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [Path(__file__), ROOT / 'tools/experiment_ego_v13.py',
               ROOT / 'tools/probe_disturbed_gate_v19.py', ROOT / 'tools/verify_development_v14.py']
    protocol = dict(purpose='Harness validation, no independent performance or risk claim',
                    model_path=model_path.relative_to(ROOT).as_posix(), model_sha256=digest(model_path),
                    expected_actual_sumo_runs=64, parity_pairs=8, shifted_runs=48,
                    shifted_scenarios=['slow_yield', 'asymmetric_speed', 'tight_headway'],
                    plants=list(PLANTS), innovation_mode='uniform', seed_base=61920001,
                    source_sha256={p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources})
    save(directory / 'protocol.json', protocol)
    for source in protocol['source_sha256']:
        destination = directory / 'source' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    rows = []
    with DisturbedMergeEnv() as original, ShiftedMergeEnv() as shifted:
        for index, regime in enumerate(('nominal', 'dense')):
            seed = 61920101 + index
            for mode in ('clean', 'uniform', 'bias', 'pulse'):
                outputs = []
                for name, env in (('original', original), ('shifted_nominal', shifted)):
                    row, trace, initial = run_episode(env, seed, regime, mode, lambda x: JointRolloutGate(), True)
                    row.update(initial=initial.tolist(), controller='baseline', engine=name)
                    rows.append(row)
                    save(directory / f'parity_{index}_{mode}_{name}.json', trace)
                    trace_audit(trace, row)
                    if name == 'shifted_nominal':
                        audit_shifted_trace(trace, row)
                    outputs.append((row, trace))
                for key in ('initial', 'per_vehicle_finish', 'per_vehicle_variation', 'completed',
                            'collision', 'contract_violations', 'aborted'):
                    np.testing.assert_allclose(outputs[0][0][key], outputs[1][0][key], atol=1e-8, rtol=0)
                for key in ('command', 'x', 'disturbance'):
                    np.testing.assert_allclose([t[key] for t in outputs[0][1]], [t[key] for t in outputs[1][1]], atol=1e-8, rtol=0)
    fs = {'baseline': lambda x: JointRolloutGate(), 'nominal_1802': lambda x: JointAdvantageGate(model)}
    for plant in PLANTS:
        with ShiftedMergeEnv(plant=plant) as env:
            for scenario in protocol['shifted_scenarios']:
                for index in range(2):
                    seed = 61920001 + index
                    for method, factory in fs.items():
                        row, trace, initial = run_episode(env, seed, scenario, 'uniform', factory, True)
                        row.update(initial=initial.tolist(), controller=method, engine='shifted')
                        rows.append(row)
                        save(directory / f'trace_{plant}_{scenario}_{index}_{method}.json', trace)
                        save(directory / 'episodes.json', rows)
                        audit_shifted_trace(trace, row)
                        if plant in ('nominal', 'gain_drag', 'lag_mild'):
                            assert row['error_contract_exceeded_substeps'] == 0
                    print(json.dumps(dict(plant=plant, scenario=scenario, index=index,
                                          actual_sumo_runs=len(rows))), flush=True)
    assert len(rows) == protocol['expected_actual_sumo_runs']
    result = dict(purpose=protocol['purpose'], actual_sumo_runs=len(rows), parity_pairs=8,
                  traces_reconstructed=len(rows), risk_certified=False,
                  plants={plant: dict(runs=sum(r.get('plant', {}).get('kind') == plant and r['engine'] == 'shifted' for r in rows),
                                      completed=sum(r['completed'] for r in rows if r.get('plant', {}).get('kind') == plant and r['engine'] == 'shifted'),
                                      collisions=sum(r['collision'] for r in rows if r.get('plant', {}).get('kind') == plant and r['engine'] == 'shifted'),
                                      aborted=sum(r['aborted'] for r in rows if r.get('plant', {}).get('kind') == plant and r['engine'] == 'shifted'),
                                      maximum_actual_error=max(r['max_actual_error'] for r in rows if r.get('plant', {}).get('kind') == plant and r['engine'] == 'shifted'),
                                      exceeded_substeps=sum(r['error_contract_exceeded_substeps'] for r in rows if r.get('plant', {}).get('kind') == plant and r['engine'] == 'shifted'))
                          for plant in PLANTS},
                  episodes_sha256=digest(directory / 'episodes.json'), protocol_sha256=digest(directory / 'protocol.json'))
    save(directory / 'summary.json', result)
    save(directory / 'sha256.json', {p.relative_to(directory).as_posix(): digest(p)
                                   for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    run(args.directory)
