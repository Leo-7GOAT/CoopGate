"""Add a frozen repair policy to the existing 128-initial development pairing.

The extension executes only the three missing controllers. Existing paired
initials are reused explicitly, never counted as new independent samples.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.advantage_gate_v18 import action_then_baseline, baseline_suffix
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.disturbed_advantage_v19 import disturbed_action_difference
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import joint_rollout
from coopgate.merge_v4 import MergeConfig
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from recover_checkpoint_v19 import atomic_bytes
from resume_frozen_risk_v19 import build_record, schedule
from review_repair_closed_loop_v20 import check_forecast_record, check_selected, outcome_counts, reconstruct_candidates
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol, regime_for
from study_repair_v20 import factories as repair_factories, trace_metadata, verify_trace
from study_temporal_v20 import is_failure, pair_delta, summarize

ROOT = Path(__file__).resolve().parents[1]
PARTNER = ROOT / 'artifacts/v20/temporal_development'
ORIGIN = ROOT / 'artifacts/v20/repair_development'
STUDY = ROOT / 'artifacts/v20/temporal_repair_extension'
PAIR_FIELDS = ('seed_base', 'n_initials', 'sampling', 'modes', 'mode_selection_stream',
               'bootstrap_stream', 'bootstrap_samples')


def methods():
    return {f'repair_{seed}': dict(kind='repair', seed=seed, bank='sampled', bank_size=33)
            for seed in (1802, 1812, 1822)}


def validate_pairing(partner, protocol):
    for key in PAIR_FIELDS:
        if partner['spec'][key] != protocol['spec'][key]:
            raise ValueError('Paired population or analysis differs: ' + key)
    if partner['models'] != protocol['models']:
        raise ValueError('Paired model identities differ')
    for source, expected in partner['source_sha256'].items():
        if protocol['source_sha256'].get(source) != expected:
            raise ValueError('Shared frozen simulator or controller changed: ' + source)
    if protocol['spec']['methods'] != methods():
        raise ValueError('Retain all three fixed repair policies')


def freeze():
    if STUDY.exists():
        raise FileExistsError('Preserve existing extension')
    _, partner = read_protocol(PARTNER)  # Immutable metadata only, no outcomes.
    _, origin = read_protocol(ORIGIN)
    verify_manifest(ORIGIN)
    review = ROOT / 'artifacts/v20/repair_closed_loop_review'
    verify_manifest(review)
    sources = dict(origin['source_sha256'])
    for source in ('tools/extend_temporal_repair_v20.py', 'tools/advance_temporal_repair_v20.py',
                   'tools/review_repair_closed_loop_v20.py', 'tests/test_temporal_repair_extension_v20.py'):
        sources[source] = digest(ROOT / source)
    spec = {key: partner['spec'][key] for key in PAIR_FIELDS}
    spec.update(purpose='Paired development extension of the frozen full-action repair policy',
                data_role='development_extension', shards=2, methods=methods(), all_traces_saved=True,
                audit_initial_indices=[], expected_actual_sumo_runs=128 * 4 * 3,
                report_scope='Report all fixed models and all existing banks; paired initial bootstrap intervals are descriptive, without risk certification.',
                timing_scope='Concurrent desktop workload; collections are not matched for timing and do not certify real-time execution.')
    protocol = dict(spec=spec, models=origin['models'], source_sha256=sources,
                    partner_protocol_sha256=digest(PARTNER / 'protocol.json'),
                    origin_protocol_sha256=digest(ORIGIN / 'protocol.json'),
                    origin_completed_review_sha256=digest(review / 'audit.json'),
                    paired_initials_reused=128, additional_independent_initials=0,
                    policy_changed_from_completed_16_initial_study=False,
                    partner_outcomes_read_for_extension_design=False,
                    v19_final_outcomes_used_to_tune_policy=False)
    validate_pairing(partner, protocol)
    # The single intentional overlap is declared. Inspect only protocol metadata.
    prior = {}
    lo, hi = spec['seed_base'], spec['seed_base'] + spec['n_initials']
    for path in sorted((ROOT / 'artifacts').rglob('protocol.json')):
        if 'source' in path.parts:
            continue
        item = json.loads(path.read_text(encoding='utf-8'))
        data = item.get('spec', item)
        start, size = data.get('seed_base'), data.get('n_initials', data.get('initials'))
        if not isinstance(start, int) or not isinstance(size, int):
            continue
        overlap = max(lo, start) < min(hi, start + size)
        if overlap and path.parent != PARTNER:
            raise ValueError('Undeclared initial overlap: ' + str(path))
        prior[path.relative_to(ROOT).as_posix()] = dict(sha256=digest(path), intentional_pairing=overlap)
    protocol['prior_protocol_ranges'] = prior
    STUDY.mkdir(parents=True)
    save(STUDY / 'protocol.json', protocol)
    for source in sources:
        target = STUDY / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    print(json.dumps(dict(expected_actual_sumo_runs=spec['expected_actual_sumo_runs'],
                          paired_initials=128, additional_independent_initials=0,
                          protocol_sha256=digest(STUDY / 'protocol.json'))), flush=True)


def worker(shard):
    torch.set_num_threads(1)
    _, protocol = read_protocol(STUDY)
    spec = protocol['spec']
    if digest(PARTNER / 'protocol.json') != protocol['partner_protocol_sha256']:
        raise ValueError('Paired protocol changed')
    require_committed([STUDY / 'protocol.json', *[ROOT / p for p in protocol['source_sha256']]])
    out = STUDY / f'shard_{shard}'
    cases = schedule(spec, shard)
    out.mkdir(exist_ok=False)
    fs = repair_factories(protocol)
    fs.pop('baseline')  # Already executed once per paired initial/mode.
    if set(fs) != set(spec['methods']):
        raise ValueError('Unexpected execution family')
    for factory in fs.values():
        factory(None).select(np.array([[-80., 12.], [-85., 13.], [-120., 11.]]), np.zeros(3))
    rows, started = [], time.perf_counter()
    with DisturbedMergeEnv() as env:
        for index, mode, name in cases:
            row, trace, initial = run_episode(env, spec['seed_base'] + index,
                                              regime_for(spec, index), mode, fs[name], True)
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


def check_grid(rows, spec):
    keys = [(r['initial_index'], r['disturbance'], r['controller']) for r in rows]
    expected = {(i, mode, name) for i in range(spec['n_initials']) for mode in spec['modes'] for name in spec['methods']}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Missing or duplicate complete paired records')


def join_rows(partner_rows, extension_rows, partner, protocol):
    validate_pairing(partner, protocol)
    check_grid(partner_rows, partner['spec'])
    check_grid(extension_rows, protocol['spec'])
    if set(partner['spec']['methods']) & set(protocol['spec']['methods']):
        raise ValueError('Controller duplicated across collections')
    spec = dict(partner['spec'])
    spec['methods'] = {**spec['methods'], **protocol['spec']['methods']}
    rows = partner_rows + extension_rows
    check_grid(rows, spec)
    return rows, spec


def paired_results(rows, spec):
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    n, modes = spec['n_initials'], spec['modes']
    draws = np.random.default_rng(spec['bootstrap_stream']).integers(n, size=(spec['bootstrap_samples'], n))
    result = {}
    for seed in (1802, 1812, 1822):
        all_gains, actual_gains, retained, added, harmed = [], [], 0, 0, 0
        for i in range(n):
            chosen = int(np.random.default_rng(np.random.SeedSequence([spec['seed_base'] + i,
                         spec['mode_selection_stream']])).integers(len(modes)))
            gains = []
            for m, mode in enumerate(modes):
                old, new, base = [lookup[i, mode, name] for name in (f'sampled_{seed}', f'repair_{seed}', 'baseline')]
                old_used, new_used = old['learned_action_steps'] > 0, new['learned_action_steps'] > 0
                if old_used and not new_used:
                    raise ValueError('Frozen expansion lost an episode adoption')
                retained += int(old_used)
                added += int(new_used and not old_used)
                harmed += int(new_used and not old_used and (is_failure(new) or np.any(pair_delta(new, base) > 1e-9)))
                gains.append(old['team_cost'] - new['team_cost'])
                if m == chosen:
                    actual_gains.append(gains[-1])
            all_gains.append(np.mean(gains))
        def estimate(values):
            values = np.asarray(values)
            return dict(mean_seconds=float(values.mean()), pointwise_bootstrap95=np.quantile(
                values[draws].mean(1), [.025, .975]).tolist())
        result[str(seed)] = dict(actual_incremental_gain=estimate(actual_gains),
                                all_mode_incremental_gain=estimate(all_gains),
                                retained_mode_adoptions=retained, added_mode_adoptions=added,
                                added_harmful_mode_runs=harmed)
    return result


def independent_review(rows, protocol):
    """Recompute accepted forecasts using the independently frozen reviewer primitives."""
    torch.set_num_threads(1)
    for item in protocol['models'].values():
        if digest(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen model changed before independent review')
    packed = parameters(MergeConfig())
    models = {seed: load_coupled(ROOT / item['path'])[0] for seed, item in protocol['models'].items()}
    _, tapes = temporal_stress_bank('sampled')
    if len(tapes) != 33:
        raise ValueError('Wrong fixed forecast bank')
    cache, counts, started = {}, Counter(), time.perf_counter()
    for number, row in enumerate(rows):
        name, index = row['controller'], row['initial_index']
        path = STUDY / f'shard_{index % protocol["spec"]["shards"]}/trace_{index}_{row["disturbance"]}_{name}.json'
        trace = json.loads(path.read_text())
        rebuilt, adopted, checks, failures = build_record(dict(row), trace, np.asarray(row['initial']), index,
                                                          name, protocol['spec']['methods'][name], protocol['spec'])
        for key in rebuilt:
            np.testing.assert_equal(rebuilt[key], row[key])
        verify_trace(trace, row, protocol['spec']['methods'][name], adopted, checks, failures)
        counts['physical_traces'] += 1
        state, previous = np.asarray(row['initial']), np.zeros(3)
        actor = JointPolicyGate(models[str(protocol['spec']['methods'][name]['seed'])])
        for step, item in enumerate(trace):
            command = np.asarray(item['command'])
            baseline = joint_rollout(state, previous, step, packed)[0]
            actor.step = step
            _, info = actor.select(state, previous)
            np.testing.assert_array_equal(item['baseline_command'], baseline)
            np.testing.assert_array_equal(item['actor_proposal'], info['proposal'])
            candidates, original_count = reconstruct_candidates(state, previous, np.asarray(info['proposal']), baseline, packed, True)
            check_selected(item, baseline, candidates, original_count)
            counts['candidate_families'] += 1
            counts['candidate_commands'] += len(candidates)
            if item['portfolio_joint_action_admitted']:
                key = (step, state.tobytes(), previous.tobytes(), command.tobytes())
                if key not in cache:
                    delta = action_then_baseline(state, previous, step, command, packed)[0] - baseline_suffix(state, previous, step, packed)[0]
                    values = [delta] + [disturbed_action_difference(state, previous, step, command, tape, packed)[0] for tape in tapes[1:]]
                    cache[key] = np.asarray(values)
                    counts['distinct_admitted_forecasts_recomputed'] += 1
                check_forecast_record(cache[key], item['selected_bank_differences'])
                counts['admitted_decisions_verified'] += 1
                counts['attenuated_decisions_verified'] += int(item['repair_action_admitted'])
            state, previous = np.asarray(item['x']), command
        if number % 40 == 0:
            print(json.dumps(dict(verified_extension_runs=number + 1, elapsed_s=time.perf_counter() - started)), flush=True)
    return dict(counts)


def finalize(logs):
    if (STUDY / 'summary.json').exists():
        raise FileExistsError('Preserve completed summary')
    logs = Path(logs)
    if json.loads((logs / 'partner_native_exit.json').read_text())['exit_code'] != 0:
        raise ValueError('Paired native supervisor did not complete successfully')
    if any(p['exit_code'] != 0 for p in json.loads((logs / 'worker_exits.json').read_text())['workers']):
        raise ValueError('Extension workers did not complete successfully')
    _, protocol = read_protocol(STUDY)
    _, partner = read_protocol(PARTNER)
    if digest(PARTNER / 'protocol.json') != protocol['partner_protocol_sha256']:
        raise ValueError('Paired protocol changed')
    verify_manifest(PARTNER)
    rows = []
    for shard in range(protocol['spec']['shards']):
        out = STUDY / f'shard_{shard}'
        receipt = json.loads((out / 'complete.json').read_text())
        if receipt['protocol_sha256'] != digest(STUDY / 'protocol.json') or receipt['episodes_sha256'] != digest(out / 'episodes.json'):
            raise ValueError('Extension shard changed')
        part = json.loads((out / 'episodes.json').read_text())
        if [(r['initial_index'], r['disturbance'], r['controller']) for r in part] != schedule(protocol['spec'], shard):
            raise ValueError('Extension schedule changed')
        if len(part) != receipt['actual_sumo_runs']:
            raise ValueError('Extension receipt count changed')
        rows.extend(part)
    check_grid(rows, protocol['spec'])
    paired = json.loads((PARTNER / 'episodes.json').read_text())
    merged, spec = join_rows(paired, rows, partner, protocol)
    results = summarize(merged, spec)
    comparisons = paired_results(merged, spec)
    counts = independent_review(rows, protocol)
    independent = {}
    for name in spec['methods']:
        independent[name] = outcome_counts(merged, spec, name)
        for key, value in independent[name].items():
            np.testing.assert_allclose(value, results[name][key], atol=1e-12, rtol=1e-12)
    rows.sort(key=lambda r: (r['initial_index'], r['disturbance'], r['controller']))
    save(STUDY / 'episodes.json', rows)
    report = dict(data_role='paired development extension', independent_initials=spec['n_initials'],
                  additional_independent_initials=0, actual_sumo_runs=len(rows), paired_existing_sumo_runs=len(paired),
                  total_unique_paired_sumo_runs=len(merged), results=results, paired_results=comparisons,
                  independent_review_counts=counts, independently_recounted_outcomes=independent,
                  risk_certified=False, hard_real_time_certified=False,
                  partner_manifest_sha256=digest(PARTNER / 'sha256.json'),
                  partner_episodes_sha256=digest(PARTNER / 'episodes.json'),
                  protocol_sha256=digest(STUDY / 'protocol.json'), episodes_sha256=digest(STUDY / 'episodes.json'),
                  interval_scope=protocol['spec']['report_scope'])
    save(STUDY / 'summary.json', report)
    save(STUDY / 'sha256.json', {p.relative_to(STUDY).as_posix(): digest(p) for p in STUDY.rglob('*')
                                if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(completed=True, actual_sumo_runs=len(rows), independent_initials=spec['n_initials'],
                          summary_sha256=digest(STUDY / 'summary.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    sub.add_parser('worker').add_argument('shard', type=int)
    sub.add_parser('finalize').add_argument('logs')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    elif args.command == 'worker':
        worker(args.shard)
    else:
        finalize(args.logs)
