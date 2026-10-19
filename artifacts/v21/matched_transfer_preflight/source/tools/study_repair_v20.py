"""Fresh paired SUMO development and exact reference checks for action repair.

The old bank, a repair-disabled adapter and the full repair policy are all run.
Every training replicate and outcome is retained. This is not risk certification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.action_repair_v20 import REPAIR_WEIGHTS
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.repair_portfolio_v20 import RepairPortfolioGate
from coopgate.temporal_portfolio_v20 import TemporalPortfolioGate
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from recover_checkpoint_v19 import atomic_bytes
from resume_frozen_risk_v19 import build_record, schedule
from start_final_v19 import require_committed
from study_gate_v19 import read_protocol, regime_for
from study_temporal_v20 import independent_ranges, summarize, verify_episode

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/repair_development'
MODEL_SEEDS = (1802, 1812, 1822)
REFERENCE_FIELDS = (
    'completed', 'collision', 'aborted', 'abort_reason', 'contract_violations',
    'initial', 'per_vehicle_finish', 'per_vehicle_cost', 'per_vehicle_variation',
    'team_cost', 'learned_action_steps', 'chosen_compositions',
    'forecast_transitions', 'internal_transitions', 'trajectory_sha256', 'decision_sha256')


def methods():
    result = {'baseline': {'kind': 'baseline'}}
    for seed in MODEL_SEEDS:
        for prefix, kind in (('sampled', 'temporal'), ('reference', 'reference'), ('repair', 'repair')):
            result[f'{prefix}_{seed}'] = dict(kind=kind, seed=seed, bank='sampled', bank_size=33)
    return result


def factories(protocol):
    result = {'baseline': lambda x: JointRolloutGate()}
    models = {}
    for seed, record in protocol['models'].items():
        if digest(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen model changed')
        model, meta = load_coupled(ROOT / record['path'])
        if meta['updates'] != meta['target_updates']:
            raise ValueError('Incomplete training')
        models[seed] = model
    for name, method in protocol['spec']['methods'].items():
        if method['kind'] == 'baseline':
            continue
        model = models[str(method['seed'])]
        if method['kind'] == 'temporal':
            result[name] = lambda x, m=model: TemporalPortfolioGate(m, 'sampled')
        elif method['kind'] in ('reference', 'repair'):
            result[name] = lambda x, m=model, repair=method['kind'] == 'repair': RepairPortfolioGate(m, 'sampled', repair)
        else:
            raise ValueError('Unknown repair-study method')
    return result


def freeze():
    if STUDY.exists():
        raise FileExistsError('Preserve the existing study')
    _, old = read_protocol(ROOT / 'artifacts/v20/temporal_development')
    spec = dict(purpose='Full-candidate action-repair closed-loop development and reference identity',
                data_role='development', seed_base=63200001, n_initials=16, shards=2,
                sampling='iid_mixture', modes=['clean', 'uniform', 'bias', 'pulse'], methods=methods(),
                all_traces_saved=True, audit_initial_indices=[], expected_actual_sumo_runs=16 * 4 * 10,
                mode_selection_stream=20307, bootstrap_stream=20313, bootstrap_samples=5000,
                repair_weights=list(REPAIR_WEIGHTS), max_failed_scenarios=0,
                bank_selection='Sampled bank chosen from completed local diagnosis only; its lower path count and identical rejection set to combined are not a closed-loop equivalence claim.',
                report_scope='All three training models and every outcome; pointwise descriptive development intervals. No risk certificate or independent final efficacy claim.',
                timing_scope='Concurrent desktop workload, no execution deadline; not an isolated or hard-real-time benchmark.',
                reference_checks='Every reference-only trajectory, common forecast evidence and work count must equal the frozen sampled controller; timings excluded.',
                adoption_checks='For each paired initial and mode, original episode adoption implies repair episode adoption; no stronger realized-performance claim.')
    sources = dict(old['source_sha256'])
    for path in ('src/coopgate/action_repair_v20.py', 'src/coopgate/repair_portfolio_v20.py',
                 'tools/study_repair_v20.py', 'tools/advance_repair_v20.py',
                 'tests/test_action_repair_v20.py', 'tests/test_repair_portfolio_v20.py',
                 'tests/test_repair_study_v20.py'):
        sources[path] = digest(ROOT / path)
    prior = independent_ranges(spec['seed_base'], spec['n_initials'])
    protocol = dict(spec=spec, models=old['models'], source_sha256=sources,
                    disjoint_prior_protocols=prior, final_confirmation_outcomes_read=False)
    if len(factories(protocol)) != 10:
        raise ValueError('All three models and both reference implementations must remain')
    STUDY.mkdir(parents=True)
    save(STUDY / 'protocol.json', protocol)
    for source in sources:
        destination = STUDY / 'source' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    print(json.dumps(dict(expected_actual_sumo_runs=spec['expected_actual_sumo_runs'],
                          disjoint_protocols=len(prior), protocol_sha256=digest(STUDY / 'protocol.json'))), flush=True)


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def trace_metadata(trace):
    trajectory = [{key: item[key] for key in ('t', 'x', 'command', 'disturbance')} for item in trace]
    decisions = [{key: item.get(key) for key in (
        'composition', 'candidate_count', 'nominal_candidate_count', 'selected_bank_differences',
        'nominal_joint_action_admitted', 'portfolio_joint_action_admitted',
        'performance_forecast_transitions', 'performance_model_transitions')} for item in trace]
    return dict(trajectory_sha256=fingerprint(trajectory), decision_sha256=fingerprint(decisions),
                repair_action_steps=sum(bool(t.get('repair_action_admitted', False)) for t in trace),
                selected_repair_weights=[t.get('repair_weight') for t in trace],
                forecast_error_count=sum(len(t.get('forecast_errors', [])) for t in trace),
                generation_error_count=sum(len(t.get('generation_errors', [])) for t in trace),
                incomplete_work_count_steps=sum(not t.get('work_counts_complete', True) for t in trace))


def verify_trace(trace, row, method, adopted, checks, failures):
    verify_episode(trace, row, method, adopted, checks, failures)
    metadata = trace_metadata(trace)
    if metadata != {key: row[key] for key in metadata}:
        raise ValueError('Trace metadata differs from archived record')
    for item in trace:
        if method['kind'] not in ('reference', 'repair'):
            continue
        used = bool(item['portfolio_joint_action_admitted'])
        weight = item['repair_weight']
        same_as_baseline = np.array_equal(item['command'], item['baseline_command'])
        if used == same_as_baseline:
            raise ValueError('Baseline execution must not count as learning adoption')
        if used and (weight not in REPAIR_WEIGHTS or (method['kind'] == 'reference' and weight != 1.)):
            raise ValueError('Unexpected selected repair weight')
        if not used and weight is not None:
            raise ValueError('A fallback cannot have a repair weight')
        if item['repair_action_admitted'] != bool(used and weight < 1.):
            raise ValueError('Repair accounting mismatch')
        if method['kind'] == 'reference' and item['candidate_count'] != item['original_candidate_count']:
            raise ValueError('Reference-only mode added candidates')
        if item['candidate_count'] < item['original_candidate_count']:
            raise ValueError('Original candidates were removed')


def worker(shard):
    torch.set_num_threads(1)
    directory, protocol = read_protocol(STUDY)
    spec = protocol['spec']
    cases = schedule(spec, shard)
    require_committed([STUDY / 'protocol.json', *[ROOT / p for p in protocol['source_sha256']]])
    out = directory / f'shard_{shard}'
    out.mkdir(exist_ok=False)
    fs, rows, started = factories(protocol), [], time.perf_counter()
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values():
        factory(warm).select(warm, np.zeros(3))
    with DisturbedMergeEnv() as env:
        for index, mode, name in cases:
            row, trace, initial = run_episode(env, spec['seed_base'] + index, regime_for(spec, index), mode, fs[name], True)
            row, adopted, checks, failures = build_record(row, trace, initial, index, name, spec['methods'][name], spec)
            row.update(trace_metadata(trace))
            save(out / f'trace_{index}_{mode}_{name}.json', trace)
            rows.append(row)
            atomic_bytes(out / 'episodes.json', (json.dumps(rows, indent=2) + '\n').encode('utf-8'))
            verify_trace(trace, row, spec['methods'][name], adopted, checks, failures)
            if len(rows) % len(fs) == 0:
                print(json.dumps(dict(shard=shard, pid=os.getpid(), initial_index=index, mode=mode,
                                      actual_sumo_runs=len(rows), expected_shard_runs=len(cases),
                                      elapsed_s=time.perf_counter() - started)), flush=True)
    save(out / 'complete.json', dict(actual_sumo_runs=len(rows), episodes_sha256=digest(out / 'episodes.json'),
                                     protocol_sha256=digest(STUDY / 'protocol.json'), pid=os.getpid()))


def compare_pairs(rows, spec):
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    expected = {(i, mode, name) for i in range(spec['n_initials']) for mode in spec['modes'] for name in spec['methods']}
    if set(lookup) != expected or len(lookup) != len(rows):
        raise ValueError('Missing or duplicate complete-study cases')
    results = {}
    for seed in MODEL_SEEDS:
        retained, added, repair_runs, harmful_added, costs, base_costs = 0, 0, 0, 0, [], []
        for i in range(spec['n_initials']):
            for mode in spec['modes']:
                original, reference, repaired, baseline = [lookup[i, mode, name] for name in
                    (f'sampled_{seed}', f'reference_{seed}', f'repair_{seed}', 'baseline')]
                for key in REFERENCE_FIELDS:
                    np.testing.assert_equal(reference[key], original[key], err_msg='Reference identity: ' + key)
                old_used, new_used = original['learned_action_steps'] > 0, repaired['learned_action_steps'] > 0
                if old_used and not new_used:
                    raise ValueError('Expanded candidate policy lost an episode adoption')
                retained += int(old_used)
                added += int(new_used and not old_used)
                repair_runs += int(repaired['repair_action_steps'] > 0)
                harm = (not repaired['completed'] or repaired['collision'] or repaired['aborted'] or repaired['contract_violations']
                        or np.any(np.asarray(repaired['per_vehicle_cost']) - baseline['per_vehicle_cost'] > 1e-9)
                        or np.any(np.asarray(repaired['per_vehicle_finish']) - baseline['per_vehicle_finish'] > 1e-9))
                harmful_added += int(new_used and not old_used and harm)
                costs.append(original['team_cost'] - repaired['team_cost'])
                base_costs.append(baseline['team_cost'])
        results[str(seed)] = dict(reference_episode_pairs=spec['n_initials'] * len(spec['modes']),
                                 retained_adopted_pairs=retained, added_adopted_pairs=added,
                                 added_harmful_pairs=harmful_added, episodes_using_attenuation=repair_runs,
                                 all_mode_mean_cost_gain_over_original=float(np.mean(costs)),
                                 all_mode_incremental_gain_percent_of_baseline=float(100 * np.mean(costs) / np.mean(base_costs)))
    return results


def finalize():
    if (STUDY / 'summary.json').exists():
        raise FileExistsError('Preserve completed analysis')
    directory, protocol = read_protocol(STUDY)
    spec, rows = protocol['spec'], []
    for shard in range(spec['shards']):
        out = directory / f'shard_{shard}'
        receipt = json.loads((out / 'complete.json').read_text())
        if receipt['protocol_sha256'] != digest(STUDY / 'protocol.json') or receipt['episodes_sha256'] != digest(out / 'episodes.json'):
            raise ValueError('Completed shard identity changed')
        part = json.loads((out / 'episodes.json').read_text())
        if [(r['initial_index'], r['disturbance'], r['controller']) for r in part] != schedule(spec, shard) or len(part) != receipt['actual_sumo_runs']:
            raise ValueError('Wrong shard schedule')
        for row in part:
            trace = json.loads((out / f"trace_{row['initial_index']}_{row['disturbance']}_{row['controller']}.json").read_text())
            rebuilt, adopted, checks, failures = build_record(dict(row), trace, np.asarray(row['initial']), row['initial_index'],
                                                              row['controller'], spec['methods'][row['controller']], spec)
            for key in rebuilt:
                np.testing.assert_equal(rebuilt[key], row[key], err_msg='Reconstructed record: ' + key)
            verify_trace(trace, row, spec['methods'][row['controller']], adopted, checks, failures)
        rows.extend(part)
    if len(rows) != spec['expected_actual_sumo_runs']:
        raise ValueError('Incomplete study')
    results, paired = summarize(rows, spec), compare_pairs(rows, spec)
    rows.sort(key=lambda r: (r['initial_index'], r['disturbance'], r['controller']))
    save(STUDY / 'episodes.json', rows)
    report = dict(purpose=spec['purpose'], data_role='development', independent_initials=spec['n_initials'],
                  actual_sumo_runs=len(rows), results=results, paired_repair_audit=paired, risk_certified=False,
                  final_confirmation_outcomes_read=False, protocol_sha256=digest(STUDY / 'protocol.json'),
                  episodes_sha256=digest(STUDY / 'episodes.json'), interval_scope=spec['report_scope'])
    save(STUDY / 'summary.json', report)
    save(STUDY / 'sha256.json', {p.relative_to(STUDY).as_posix(): digest(p) for p in STUDY.rglob('*')
                                if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    p = sub.add_parser('worker')
    p.add_argument('shard', type=int)
    sub.add_parser('finalize')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    elif args.command == 'worker':
        worker(args.shard)
    else:
        finalize()
