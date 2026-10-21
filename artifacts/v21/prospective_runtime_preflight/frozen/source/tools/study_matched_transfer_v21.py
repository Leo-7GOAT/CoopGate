"""Frozen transfer comparison with identical external noise across every plant."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.advantage_gate_v18 import JointAdvantageGate, action_then_baseline, baseline_suffix
from coopgate.arrival_phase_v20 import arrival_phase_suffix
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import JointRolloutGate, joint_rollout
from coopgate.matched_shift_v21 import INNOVATION_RADIUS, MatchedShiftedMergeEnv, audit_matched_trace
from coopgate.merge_v4 import MergeConfig
from coopgate.phase_repair_v21 import PhaseRepairGate
from coopgate.portfolio_gate_v19 import JointPortfolioGate
from coopgate.repair_portfolio_v20 import RepairPortfolioGate
from coopgate.robust_v9 import robust_certificate
from coopgate.shifted_sumo_v19 import PLANTS, SCENARIOS, shifted_initial
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from recover_checkpoint_v19 import atomic_bytes
from resume_frozen_risk_v19 import build_record
from review_repair_closed_loop_v20 import check_forecast_record, check_selected, reconstruct_candidates
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol
from study_phase_repair_v21 import MODELS, phase_record_check, validate_worker_exits
from study_repair_v20 import trace_metadata
from study_temporal_v20 import is_failure

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v21/matched_transfer'
PREFLIGHT = ROOT / 'artifacts/v21/matched_transfer_preflight'
NEW_SOURCES = ('src/coopgate/matched_shift_v21.py', 'tools/study_matched_transfer_v21.py',
               'tools/preflight_matched_transfer_v21.py', 'tools/advance_matched_transfer_v21.py',
               'tests/test_matched_shift_v21.py', 'tests/test_matched_transfer_v21.py')


def methods(include_reference=False):
    result = {'baseline': dict(kind='baseline')}
    for seed in MODELS:
        result[f'raw_{seed}'] = dict(kind='raw', seed=seed)
        result[f'nominal_{seed}'] = dict(kind='nominal', seed=seed)
        result[f'portfolio0_{seed}'] = dict(kind='portfolio', seed=seed, bank_size=17,
                                          max_failed_scenarios=0)
        for name, variant, tolerance in (('repair', 'original', None), ('phase05', 'phase', .05),
                                         ('phase10', 'phase', .1)):
            result[f'{name}_{seed}'] = dict(kind='repair', seed=seed, variant=variant,
                bank='sampled', bank_size=33, phase_tolerance_seconds=tolerance)
        if include_reference:
            result[f'reference_{seed}'] = dict(kind='repair', seed=seed, variant='phase',
                bank='sampled', bank_size=33, phase_tolerance_seconds=None)
    return result


def factories(protocol):
    models, result = {}, {}
    for seed, record in protocol['models'].items():
        if digest(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen training model changed')
        model, meta = load_coupled(ROOT / record['path'])
        if meta['updates'] != meta['target_updates']:
            raise ValueError('Incomplete training model')
        models[seed] = model
    for name, method in protocol['spec']['methods'].items():
        kind = method['kind']
        if kind == 'baseline':
            result[name] = lambda x: JointRolloutGate()
            continue
        model = models[str(method['seed'])]
        if kind == 'raw': result[name] = lambda x, m=model: JointPolicyGate(m)
        elif kind == 'nominal': result[name] = lambda x, m=model: JointAdvantageGate(m)
        elif kind == 'portfolio': result[name] = lambda x, m=model: JointPortfolioGate(m, 0)
        elif kind == 'repair' and method['variant'] == 'original':
            result[name] = lambda x, m=model: RepairPortfolioGate(m, 'sampled', True)
        elif kind == 'repair' and method['variant'] == 'phase':
            result[name] = lambda x, m=model, eta=method['phase_tolerance_seconds']: PhaseRepairGate(m, eta)
        else: raise ValueError('Unknown fixed transfer controller')
    return result


def cases(spec):
    count = spec['n_initials_per_scenario']
    if not 1 <= count <= 1000:
        raise ValueError('Scenario seed blocks would overlap or be empty')
    for si, scenario in enumerate(spec['scenarios']):
        for index in range(count):
            yield si * count + index, scenario, index, spec['seed_base'] + si * 1000 + index


def schedule(spec, shard):
    if not 0 <= shard < spec['shards']:
        raise ValueError('Unknown shard')
    result = []
    for global_index, scenario, index, seed in cases(spec):
        if global_index % spec['shards'] != shard: continue
        for pi, plant in enumerate(spec['plants']):
            order = np.random.default_rng(np.random.SeedSequence([seed, pi, 21209])).permutation(list(spec['methods']))
            result.extend((global_index, scenario, index, seed, plant, str(name)) for name in order)
    return result


def identity(row):
    return (row['global_initial_index'], row['scenario'], row['initial_index'], row['seed'],
            row['plant_name'], row['controller'])


def record(row, trace, initial, case, spec):
    global_index, scenario, index, seed, plant, name = case
    if row['seed'] != seed or row['scenario'] != scenario:
        raise ValueError('Simulator executed a different initial')
    row, _, _, _ = build_record(row, trace, initial, index, name, spec['methods'][name], spec)
    row.update(global_initial_index=global_index, plant_name=plant, **trace_metadata(trace))
    return row


def verify_record(trace, row, method):
    physical = audit_matched_trace(trace, row)
    if trace_metadata(trace) != {k: row[k] for k in trace_metadata(trace)}:
        raise ValueError('Archived trajectory/decision metadata differs')
    for item in trace:
        admitted = bool(item.get('portfolio_joint_action_admitted', False))
        if admitted:
            bank = np.asarray(item['selected_bank_differences'])
            if (bank.shape != (method['bank_size'], 3, 2) or not np.isfinite(bank).all()
                    or np.any(bank > 1e-10) or bank[0, :, 0].mean() >= -1e-9):
                raise ValueError('Incomplete/invalid admitted performance bank')
        if method['kind'] != 'repair': continue
        phase_record_check(item, method)
        if item['temporal_bank'] != 'sampled': raise ValueError('Different performance bank')
        same = np.array_equal(item['command'], item['baseline_command'])
        weight = item['repair_weight']
        if admitted == same or (admitted and weight not in (1., .75, .5, .25)):
            raise ValueError('Incorrect learning adoption or repair weight')
        if (not admitted and weight is not None) or item['repair_action_admitted'] != bool(admitted and weight < 1.):
            raise ValueError('Incorrect repair accounting')
        if item['candidate_count'] < item['original_candidate_count']:
            raise ValueError('Original candidate removed')
    return physical


def validate_grid(rows, spec):
    lookup = {identity(r): r for r in rows}
    expected = {entry for shard in range(spec['shards']) for entry in schedule(spec, shard)}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError('Missing, duplicated or substituted transfer case')
    return {(r['scenario'], r['initial_index'], r['plant_name'], r['controller']): r for r in rows}


def summarize(rows, spec):
    lookup = validate_grid(rows, spec)
    n = spec['n_initials_per_scenario']
    draws = np.random.default_rng(spec['bootstrap_stream']).integers(n, size=(spec['bootstrap_samples'], n))
    cells, plant_contrasts, gate_contrasts, zero_adoption_checks = [], [], [], 0
    adoption_subset_checks = 0
    for _, scenario, index, seed in cases(spec):
        common = lookup[scenario, index, spec['plants'][0], 'baseline']
        for plant in spec['plants']:
            base = lookup[scenario, index, plant, 'baseline']
            for name, method in spec['methods'].items():
                row = lookup[scenario, index, plant, name]
                np.testing.assert_array_equal(row['initial'], common['initial'])
                if (row['seed'] != seed or row['innovation_tape_sha256'] != common['innovation_tape_sha256'] or
                        row['external_innovation_radius'] != INNOVATION_RADIUS or row['plant'] != base['plant']):
                    raise ValueError('Paired external inputs/plant parameters differ')
                if method['kind'] not in ('baseline', 'raw') and row['learned_action_steps'] == 0:
                    for key in ('per_vehicle_cost', 'per_vehicle_finish', 'completed', 'aborted', 'collision',
                                'contract_violations', 'trajectory_sha256', 'max_actual_error'):
                        np.testing.assert_equal(row[key], base[key], err_msg='Zero-adoption baseline identity: ' + key)
                    zero_adoption_checks += 1
            for seed_id in MODELS:
                group = [lookup[scenario, index, plant, f'{prefix}_{seed_id}']
                         for prefix in ('phase05', 'phase10', 'repair')]
                if not any(r['forecast_error_count'] or r['generation_error_count'] for r in group):
                    used = [r['learned_action_steps'] > 0 for r in group]
                    if used != sorted(used): raise ValueError('A stricter eligible set created an episode adoption')
                    adoption_subset_checks += 1
    gains_by_cell = {}
    for scenario in spec['scenarios']:
        for plant in spec['plants']:
            for name in spec['methods']:
                data = [lookup[scenario, i, plant, name] for i in range(n)]
                bases = [lookup[scenario, i, plant, 'baseline'] for i in range(n)]
                gains = np.asarray([b['team_cost'] - r['team_cost'] for r, b in zip(data, bases)])
                harmful = [bool(is_failure(r) or np.any(np.asarray(r['per_vehicle_cost']) - b['per_vehicle_cost'] > 1e-9)
                                or np.any(np.asarray(r['per_vehicle_finish']) - b['per_vehicle_finish'] > 1e-9))
                           for r, b in zip(data, bases)]
                adopted = np.asarray([r['learned_action_steps'] > 0 for r in data])
                times = [t for row in data for t in row['controller_ms']]
                gains_by_cell[scenario, plant, name] = gains
                cells.append(dict(scenario=scenario, plant=plant, method=name, independent_initials=n,
                    mean_gain_seconds=float(gains.mean()), gain_pointwise_bootstrap95=np.quantile(gains[draws].mean(1), [.025, .975]).tolist(),
                    mean_gain_percent=float(100 * gains.mean() / np.mean([b['team_cost'] for b in bases])),
                    adopted=int(adopted.sum()), harmful=int(sum(harmful)), harmful_adopted=int(np.sum(adopted & harmful)),
                    completed=sum(r['completed'] for r in data), aborted=sum(r['aborted'] for r in data),
                    collisions=sum(r['collision'] for r in data), contract_violations=sum(r['contract_violations'] for r in data),
                    baseline_failed_runs=sum(is_failure(b) for b in bases),
                    maximum_actual_error=max(r['max_actual_error'] for r in data),
                    exceeded_error_runs=sum(r['error_contract_exceeded_substeps'] > 0 for r in data),
                    forecast_error_count=sum(r['forecast_error_count'] for r in data),
                    generation_error_count=sum(r['generation_error_count'] for r in data),
                    controller_p95_ms=float(np.percentile(times, 95)) if times else None,
                    controller_p99_ms=float(np.percentile(times, 99)) if times else None,
                    steps_over_500ms=sum(t > 500 for t in times), control_steps=len(times)))
            for seed in MODELS:
                for prefix in ('phase05', 'phase10'):
                    name = f'{prefix}_{seed}'
                    contrast = gains_by_cell[scenario, plant, name] - gains_by_cell[scenario, plant, f'repair_{seed}']
                    gate_contrasts.append(dict(scenario=scenario, plant=plant, method=name,
                        reference=f'repair_{seed}', independent_initials=n, mean_increment_seconds=float(contrast.mean()),
                        pointwise_bootstrap95=np.quantile(contrast[draws].mean(1), [.025, .975]).tolist()))
        for plant in spec['plants']:
            if plant == 'nominal': continue
            for name in spec['methods']:
                contrast = gains_by_cell[scenario, plant, name] - gains_by_cell[scenario, 'nominal', name]
                plant_contrasts.append(dict(scenario=scenario, plant=plant, method=name,
                    quantity='Paired change in baseline-relative gain caused by the declared plant intervention at fixed external noise',
                    mean_seconds=float(contrast.mean()), pointwise_bootstrap95=np.quantile(contrast[draws].mean(1), [.025, .975]).tolist(),
                    independent_initials=n))
    return dict(cells=cells, paired_plant_contrasts=plant_contrasts, paired_gate_contrasts=gate_contrasts,
                zero_adoption_baseline_checks=zero_adoption_checks,
                error_free_adoption_subset_checks=adoption_subset_checks)


def source_records():
    _, phase = read_protocol(ROOT / 'artifacts/v21/phase_development')
    sources = dict(phase['source_sha256'])
    for path in (ROOT / 'src/coopgate').glob('*.py'): sources[path.relative_to(ROOT).as_posix()] = digest(path)
    for name in NEW_SOURCES: sources[name] = digest(ROOT / name)
    return sources, phase['models']


def disjoint_protocols(spec):
    requested = {seed for _, _, _, seed in cases(spec)}
    output = {}
    for path in sorted((ROOT / 'artifacts').rglob('protocol.json')):
        if 'source' in path.parts or path.parent == STUDY: continue
        old = json.loads(path.read_text(encoding='utf-8'))
        old = old.get('spec', old)
        start = old.get('seed_base')
        if not isinstance(start, int): continue
        if isinstance(old.get('n_initials_per_scenario'), int):
            ranges = [(start + si * 1000, old['n_initials_per_scenario']) for si in range(len(old['scenarios']))]
        else:
            count = old.get('n_initials', old.get('initials'))
            if not isinstance(count, int): continue
            ranges = [(start, count)]
        if any(start <= seed < start + count for start, count in ranges for seed in requested):
            raise ValueError('Fresh transfer seeds overlap: ' + str(path))
        output[path.relative_to(ROOT).as_posix()] = dict(ranges=ranges, sha256=digest(path))
    return output


def freeze():
    if STUDY.exists(): raise FileExistsError('Preserve the existing matched transfer study')
    verify_manifest(PREFLIGHT)
    checked = json.loads((PREFLIGHT / 'audit.json').read_text())
    if not checked['all_checks_pass'] or checked['actual_sumo_runs'] != checked['expected_actual_sumo_runs']:
        raise ValueError('Actual matched transfer integration has not passed')
    sources, models = source_records()
    preflight_sources = json.loads((PREFLIGHT / 'plan.json').read_text())['source_sha256']
    if sources != preflight_sources:
        raise ValueError('Implementation changed after matched preflight')
    spec = dict(purpose='Matched-noise direct transfer of every fixed model and both phase gates',
        data_role='direct_transfer_development', seed_base=63400001, n_initials_per_scenario=32,
        scenarios=list(SCENARIOS), plants=list(PLANTS), mode='uniform', external_innovation_radius=INNOVATION_RADIUS,
        physical_error_radius=.5, shards=3, methods=methods(), all_traces_saved=True, audit_initial_indices=[],
        expected_actual_sumo_runs=5 * 32 * 4 * 19, bootstrap_stream=21213, bootstrap_samples=5000,
        risk_certified=False, weights_or_gates_recalibrated=False,
        report_scope='32 independent initials per scenario; plants and controllers are paired, not independent samples. Pointwise descriptive intervals only; all failures retained.',
        failure_scope='Incomplete vehicle arrival capped at 25s; no safety claim about motion after a simulation abort.',
        selection_basis='Both fixed phase margins and all models, chosen before reading V21 phase-study outcomes; no winner selection.',
        timing_scope='Concurrent desktop workload; no hard-real-time or isolated hardware timing claim.')
    prior = disjoint_protocols(spec)
    packed = parameters(MergeConfig())
    for _, scenario, _, seed in cases(spec):
        if scenario not in ('nominal', 'dense') and not robust_certificate(
                shifted_initial(seed, scenario), np.zeros(3), np.full(3, -2.5), packed, .5):
            raise ValueError('New initial outside original recoverable set; no seed replacement')
    protocol = dict(spec=spec, models=models, source_sha256=sources, disjoint_prior_protocols=prior,
        preflight_manifest_sha256=digest(PREFLIGHT / 'sha256.json'), phase_study_outcomes_read=False)
    if len(factories(protocol)) != 19: raise ValueError('Every controller/model must remain')
    STUDY.mkdir(parents=True)
    save(STUDY / 'protocol.json', protocol)
    for name in sources:
        target = STUDY / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    print(json.dumps(dict(actual_runs_planned=spec['expected_actual_sumo_runs'], independent_initials=160,
                         protocol_sha256=digest(STUDY / 'protocol.json'))), flush=True)


def worker(shard):
    torch.set_num_threads(1)
    _, protocol = read_protocol(STUDY)
    require_committed([STUDY / 'protocol.json', *[ROOT / name for name in protocol['source_sha256']]])
    spec, rows = protocol['spec'], []
    out = STUDY / f'shard_{shard}'
    out.mkdir(exist_ok=False)
    fs, assigned, started = factories(protocol), schedule(spec, shard), time.perf_counter()
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values(): factory(warm).select(warm, np.zeros(3))
    with ExitStack() as stack:
        environments = {p: stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in spec['plants']}
        for case in assigned:
            global_index, scenario, index, seed, plant, name = case
            row, trace, initial = run_episode(environments[plant], seed, scenario, spec['mode'], fs[name], True)
            row = record(row, trace, initial, case, spec)
            save(out / f'trace_{global_index}_{plant}_{name}.json', trace)
            rows.append(row)
            atomic_bytes(out / 'episodes.json', (json.dumps(rows, indent=2) + '\n').encode('utf-8'))
            verify_record(trace, row, spec['methods'][name])
            if len(rows) % len(fs) == 0:
                print(json.dumps(dict(shard=shard, pid=os.getpid(), global_initial_index=global_index,
                    scenario=scenario, plant=plant, actual_sumo_runs=len(rows), expected_shard_runs=len(assigned),
                    elapsed_s=time.perf_counter() - started)), flush=True)
    save(out / 'complete.json', dict(pid=os.getpid(), actual_sumo_runs=len(rows),
        episodes_sha256=digest(out / 'episodes.json'), protocol_sha256=digest(STUDY / 'protocol.json')))


def independent_review(rows, protocol, trace_loader):
    torch.set_num_threads(1)
    models = {}
    for seed, entry in protocol['models'].items():
        if digest(ROOT / entry['path']) != entry['sha256']: raise ValueError('Model changed before review')
        models[seed] = load_coupled(ROOT / entry['path'])[0]
    packed, cache = parameters(MergeConfig()), {}
    _, tapes = temporal_stress_bank('sampled')
    counts = dict(physical_traces=0, candidate_families=0, distinct_admitted_forecasts=0,
                  admitted_forecasts=0, raw_actor_checks=0, nominal_checks=0, error_case_controller_replays=0)
    error_cases, replay_factories = [], None
    for number, row in enumerate(rows):
        trace, method = trace_loader(row), protocol['spec']['methods'][row['controller']]
        rebuilt = record(dict(row), trace, np.asarray(row['initial']), identity(row), protocol['spec'])
        for key in rebuilt: np.testing.assert_equal(row[key], rebuilt[key])
        verify_record(trace, row, method)
        counts['physical_traces'] += 1
        actor = JointPolicyGate(models[str(method['seed'])]) if method['kind'] != 'baseline' else None
        state, previous = np.asarray(row['initial']), np.zeros(3)
        for step, item in enumerate(trace):
            command = np.asarray(item['command'])
            kind = method['kind']
            if kind in ('raw', 'repair'):
                actor.step = step
                raw_command, actor_info = actor.select(state, previous)
                if kind == 'raw':
                    np.testing.assert_array_equal(command, raw_command)
                    counts['raw_actor_checks'] += 1
                else:
                    baseline = joint_rollout(state, previous, step, packed)[0]
                    np.testing.assert_array_equal(item['actor_proposal'], actor_info['proposal'])
                    np.testing.assert_array_equal(item['baseline_command'], baseline)
                    if item['forecast_errors'] or item['generation_errors']:
                        # Exceptions are retained and exactly replayed. These are
                        # not counted as independent candidate-family audits.
                        if replay_factories is None: replay_factories = factories(protocol)
                        controller = replay_factories[row['controller']](state)
                        controller.step = step
                        replay_command, info = controller.select(state, previous)
                        np.testing.assert_array_equal(command, replay_command)
                        for field in ('forecast_errors', 'generation_errors', 'composition', 'repair_weight',
                                      'selected_bank_differences', 'portfolio_joint_action_admitted'):
                            np.testing.assert_equal(item[field], info[field])
                        error_cases.append(dict(case=identity(row), step=step,
                            forecast_errors=item['forecast_errors'], generation_errors=item['generation_errors']))
                        counts['error_case_controller_replays'] += 1
                    else:
                        family, original_count = reconstruct_candidates(state, previous, np.asarray(actor_info['proposal']), baseline, packed, True)
                        check_selected(item, baseline, family, original_count)
                        counts['candidate_families'] += 1
            if kind == 'nominal' and item.get('nominal_joint_action_admitted', False):
                base = baseline_suffix(state, previous, step, packed)[0]
                value = action_then_baseline(state, previous, step, command, packed)[0]
                delta = value - base
                if np.any(delta > 1e-10) or delta[:, 0].mean() >= -1e-9:
                    raise ValueError('Reconstructed nominal admission fails')
                counts['nominal_checks'] += 1
            if item.get('portfolio_joint_action_admitted', False):
                bank_size = method['bank_size']
                key = (bank_size, step, state.tobytes(), previous.tobytes(), command.tobytes())
                if key not in cache:
                    differences, phases, complete = [], [], True
                    for tape in tapes[:bank_size]:
                        base, bt, bc = arrival_phase_suffix(state, previous, step, tape, packed, command, False)
                        value, vt, vc = arrival_phase_suffix(state, previous, step, tape, packed, command, True)
                        differences.append(value - base)
                        phases.append(vt - bt)
                        complete = complete and bc and vc
                    cache[key] = np.asarray(differences), np.asarray(phases), complete
                    counts['distinct_admitted_forecasts'] += 1
                differences, phases, complete = cache[key]
                if kind == 'repair': check_forecast_record(differences, item['selected_bank_differences'])
                elif kind == 'portfolio':
                    np.testing.assert_allclose(differences[:17], item['selected_bank_differences'], atol=1e-10, rtol=0)
                else: raise ValueError('Unknown portfolio admission')
                if method.get('phase_tolerance_seconds') is not None:
                    if not complete: raise ValueError('Admitted phase forecast incomplete')
                    np.testing.assert_allclose(phases, item['selected_phase_differences'], atol=1e-10, rtol=0)
                    if np.any(phases > method['phase_tolerance_seconds'] + 1e-10):
                        raise ValueError('Reconstructed phase admission fails')
                counts['admitted_forecasts'] += 1
            state, previous = np.asarray(item['x']), command
        if number % 40 == 0:
            print(json.dumps(dict(verified_runs=number + 1, total_runs=len(rows))), flush=True)
    return dict(**counts, retained_error_cases=error_cases)


def finalize(logs):
    if (STUDY / 'summary.json').exists(): raise FileExistsError('Preserve the completed analysis')
    _, protocol = read_protocol(STUDY)
    launch = json.loads((Path(logs) / 'launch.json').read_text())
    exits = json.loads((Path(logs) / 'worker_exits.json').read_text())['workers']
    if launch['protocol_sha256'] != digest(STUDY / 'protocol.json'): raise ValueError('Launch protocol changed')
    spec, rows = protocol['spec'], []
    validate_worker_exits(exits, launch, spec['shards'])
    for shard in range(spec['shards']):
        out = STUDY / f'shard_{shard}'
        receipt = json.loads((out / 'complete.json').read_text())
        if receipt['protocol_sha256'] != digest(STUDY / 'protocol.json') or receipt['episodes_sha256'] != digest(out / 'episodes.json'):
            raise ValueError('Completed transfer shard changed')
        part = json.loads((out / 'episodes.json').read_text())
        if [identity(row) for row in part] != schedule(spec, shard) or len(part) != receipt['actual_sumo_runs']:
            raise ValueError('Incomplete transfer schedule')
        rows.extend(part)
    if len(rows) != spec['expected_actual_sumo_runs']: raise ValueError('Incomplete transfer study')
    analysis = summarize(rows, spec)
    def loader(row):
        return json.loads((STUDY / f"shard_{row['global_initial_index'] % spec['shards']}" /
            f"trace_{row['global_initial_index']}_{row['plant_name']}_{row['controller']}.json").read_text())
    reviewed = independent_review(rows, protocol, loader)
    rows.sort(key=identity)
    save(STUDY / 'episodes.json', rows)
    save(STUDY / 'summary.json', dict(actual_sumo_runs=len(rows), independent_initials=5 * spec['n_initials_per_scenario'],
        **analysis, independent_review_counts=reviewed, risk_certified=False, hard_real_time_certified=False,
        protocol_sha256=digest(STUDY / 'protocol.json'), episodes_sha256=digest(STUDY / 'episodes.json')))
    save(STUDY / 'sha256.json', {p.relative_to(STUDY).as_posix(): digest(p) for p in STUDY.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(completed=True, actual_sumo_runs=len(rows))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    sub.add_parser('worker').add_argument('shard', type=int)
    sub.add_parser('finalize').add_argument('logs')
    args = parser.parse_args()
    if args.command == 'freeze': freeze()
    elif args.command == 'worker': worker(args.shard)
    else: finalize(args.logs)
