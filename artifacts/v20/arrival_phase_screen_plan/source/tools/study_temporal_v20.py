"""Fresh paired SUMO development of fixed temporal-bank extensions.

All three models and all three banks are compared on identical exogenous tapes.
This split is for development, never independent risk certification.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.portfolio_gate_v19 import JointPortfolioGate
from coopgate.temporal_portfolio_v20 import BANKS, TemporalPortfolioGate, temporal_stress_bank
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from recover_checkpoint_v19 import atomic_bytes
from resume_frozen_risk_v19 import build_record, schedule
from start_final_v19 import require_committed
from study_gate_v19 import read_protocol, regime_for
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/temporal_development'


def methods():
    result = {'baseline': {'kind': 'baseline'}}
    for seed in (1802, 1812, 1822):
        result[f'portfolio0_{seed}'] = dict(kind='portfolio', seed=seed, bank='original', bank_size=17)
        for bank in BANKS:
            result[f'{bank}_{seed}'] = dict(kind='temporal', seed=seed, bank=bank,
                                           bank_size=len(temporal_stress_bank(bank)[0]))
    return result


def factories(protocol):
    fs = {'baseline': lambda x: JointRolloutGate()}
    models = {}
    for seed, record in protocol['models'].items():
        if digest(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen model changed')
        model, meta = load_coupled(ROOT / record['path'])
        if meta['updates'] != meta['target_updates']:
            raise ValueError('Incomplete training')
        models[seed] = model
    for name, spec in protocol['spec']['methods'].items():
        if spec['kind'] == 'baseline':
            continue
        model = models[str(spec['seed'])]
        if spec['kind'] == 'portfolio':
            fs[name] = lambda x, m=model: JointPortfolioGate(m, 0)
        elif spec['kind'] == 'temporal':
            fs[name] = lambda x, m=model, bank=spec['bank']: TemporalPortfolioGate(m, bank)
        else:
            raise ValueError('Unknown fixed method')
    return fs


def independent_ranges(seed_base, count):
    """Read immutable protocol metadata only, never final-test outcomes."""
    records = {}
    for path in sorted((ROOT / 'artifacts').rglob('protocol.json')):
        if 'source' in path.parts or path.parent == STUDY:
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        spec = data.get('spec', data)
        start, size = spec.get('seed_base'), spec.get('n_initials', spec.get('initials'))
        if not isinstance(start, int) or not isinstance(size, int):
            continue
        if max(start, seed_base) < min(start + size, seed_base + count):
            raise ValueError('Initial range overlaps an existing protocol: ' + str(path))
        records[path.relative_to(ROOT).as_posix()] = dict(seed_base=start, initials=size, sha256=digest(path))
    return records


def freeze():
    if STUDY.exists():
        raise FileExistsError('Preserve the existing study')
    _, old = read_protocol(ROOT / 'artifacts/v19/risk_validation')
    spec = dict(purpose='Fresh closed-loop temporal-bank development, not risk certification',
                data_role='development', seed_base=63100001, n_initials=128, shards=2,
                sampling='iid_mixture', modes=['clean', 'uniform', 'bias', 'pulse'],
                methods=methods(), all_traces_saved=False, audit_initial_indices=[0, 1, 64, 65],
                expected_actual_sumo_runs=128 * 4 * 13, mode_selection_stream=20207,
                bootstrap_stream=20213, bootstrap_samples=5000,
                report_scope='Report every bank and training model. Development intervals are pointwise descriptive estimates; no post-selection risk or final-efficacy claim.',
                timing_scope='Concurrent desktop experiments; measured timings describe this collection, not an isolated deployment benchmark.')
    sources = dict(old['source_sha256'])
    for path in ('src/coopgate/temporal_portfolio_v20.py', 'tools/study_temporal_v20.py',
                 'tests/test_temporal_study_v20.py', 'tools/resume_frozen_risk_v19.py',
                 'tools/recover_checkpoint_v19.py', 'tools/start_final_v19.py'):
        sources[path] = digest(ROOT / path)
    prior = independent_ranges(spec['seed_base'], spec['n_initials'])
    protocol = dict(spec=spec, models=old['models'], source_sha256=sources,
                    disjoint_prior_protocols=prior, final_confirmation_outcomes_read=False)
    if len(factories(protocol)) != 13:
        raise ValueError('All three models and four banks must remain')
    STUDY.mkdir(parents=True)
    save(STUDY / 'protocol.json', protocol)
    for source in sources:
        target = STUDY / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    print(json.dumps(dict(expected_actual_sumo_runs=spec['expected_actual_sumo_runs'],
                          disjoint_protocols=len(prior), protocol_sha256=digest(STUDY / 'protocol.json'))), flush=True)


def verify_episode(trace, row, method_spec, adopted, checks, failures):
    trace_audit(trace, row)
    if adopted and (any(count != method_spec['bank_size'] for count in checks) or any(failures)):
        raise ValueError('A joint action was admitted without passing every declared path')
    for item in trace:
        if item.get('temporal_bank') is not None and item['temporal_bank'] != method_spec['bank']:
            raise ValueError('Executed a different temporal bank')


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
            row, trace, initial = run_episode(env, spec['seed_base'] + index,
                                              regime_for(spec, index), mode, fs[name], True)
            row, adopted, checks, failures = build_record(row, trace, initial, index, name, spec['methods'][name], spec)
            # Keep the original simulator's failure record before verification.
            if row['trace_saved']:
                save(out / f'trace_{index}_{mode}_{name}.json', trace)
            rows.append(row)
            atomic_bytes(out / 'episodes.json', (json.dumps(rows, indent=2) + '\n').encode('utf-8'))
            verify_episode(trace, row, spec['methods'][name], adopted, checks, failures)
            if len(rows) % len(fs) == 0:
                print(json.dumps(dict(shard=shard, pid=os.getpid(), initial_index=index, mode=mode,
                                      actual_sumo_runs=len(rows), expected_shard_runs=len(cases),
                                      elapsed_s=time.perf_counter() - started)), flush=True)
    save(out / 'complete.json', dict(actual_sumo_runs=len(rows), episodes_sha256=digest(out / 'episodes.json'),
                                     protocol_sha256=digest(STUDY / 'protocol.json'), pid=os.getpid()))


def pair_delta(row, base):
    return np.column_stack((row['per_vehicle_cost'], row['per_vehicle_finish'])) - np.column_stack(
        (base['per_vehicle_cost'], base['per_vehicle_finish']))


def is_failure(row):
    return bool(not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations'])


def summarize(rows, spec):
    lookup = {(r['initial_index'], r['disturbance'], r['controller']): r for r in rows}
    expected = {(i, mode, name) for i in range(spec['n_initials']) for mode in spec['modes'] for name in spec['methods']}
    if set(lookup) != expected or len(lookup) != len(rows):
        raise ValueError('Missing or duplicate complete-study cases')
    n, modes = spec['n_initials'], spec['modes']
    chosen = [int(np.random.default_rng(np.random.SeedSequence([spec['seed_base'] + i,
              spec['mode_selection_stream']])).integers(len(modes))) for i in range(n)]
    resamples = np.random.default_rng(spec['bootstrap_stream']).integers(n, size=(spec['bootstrap_samples'], n))
    results = {}
    for name, method in spec['methods'].items():
        gain, base_cost = np.zeros((n, len(modes))), np.zeros((n, len(modes)))
        adoption, harm = np.zeros_like(gain, dtype=bool), np.zeros_like(gain, dtype=bool)
        failures, timings, losses = 0, [], []
        strata = {}
        for i in range(n):
            for m, mode in enumerate(modes):
                row, base = lookup[i, mode, name], lookup[i, mode, 'baseline']
                if row['seed'] != spec['seed_base'] + i or row['scenario'] != regime_for(spec, i) or row['scenario'] != base['scenario']:
                    raise ValueError('Initial identity differs from frozen population')
                np.testing.assert_allclose(row['initial'], base['initial'], atol=1e-8, rtol=0)
                delta = pair_delta(row, base)
                used = row['learned_action_steps'] > 0
                failed = is_failure(row)
                if not used:
                    np.testing.assert_allclose(delta, 0., atol=1e-9, rtol=0)
                    if failed != is_failure(base):
                        raise ValueError('Zero-adoption failure differs from baseline')
                if used:
                    if row['min_adopted_bank_checks'] != method['bank_size'] or row['max_adopted_failed_scenarios'] != 0:
                        raise ValueError('Missing full-bank admission evidence')
                # These two actual modes belong to every fixed bank.
                if mode in ('clean', 'bias') and (failed or np.any(delta > 1e-9)):
                    raise ValueError('In-bank completed-trajectory contract failed')
                gain[i, m], base_cost[i, m] = -delta[:, 0].mean(), base['team_cost']
                adoption[i, m], harm[i, m] = used, failed or np.any(delta > 1e-9)
                failures += int(failed)
                timings.extend(row['controller_ms'])
                losses.append(float(delta.max()))
        select = (np.arange(n), np.asarray(chosen))
        g, b, a, h = gain[select], base_cost[select], adoption[select], harm[select]
        interval = np.quantile(g[resamples].mean(axis=1), [.025, .975]).tolist()
        for regime in ('nominal', 'dense'):
            ids = [i for i in range(n) if regime_for(spec, i) == regime]
            strata[regime] = dict(initials=len(ids), adopted=int(a[ids].sum()), harmful_adopted=int((a[ids] & h[ids]).sum()),
                                  mean_gain_percent=float(100 * g[ids].mean() / b[ids].mean()) if ids else None)
        results[name] = dict(actual_adopted=int(a.sum()), actual_adoption_rate=float(a.mean()),
                             actual_harmful_adopted=int((a & h).sum()),
                             actual_empirical_conditional_harm=float((a & h).sum() / a.sum()) if a.any() else None,
                             actual_mean_gain=float(g.mean()), actual_mean_gain_percent=float(100 * g.mean() / b.mean()),
                             actual_mean_gain_pointwise_bootstrap95=interval,
                             bundle_adopted=int(adoption.any(1).sum()), bundle_harmful_adopted=int((adoption.any(1) & harm.any(1)).sum()),
                             all_mode_mean_gain_percent=float(100 * gain.mean() / base_cost.mean()),
                             harmful_runs=int(harm.sum()), completed=n * len(modes) - failures,
                             failed_runs=failures, runs=n * len(modes), worst_component_loss=max(losses),
                             controller_p95_ms=float(np.percentile(timings, 95)), controller_p99_ms=float(np.percentile(timings, 99)),
                             controller_max_ms=float(max(timings)), controller_steps=len(timings),
                             deadline_misses_500ms=int(np.count_nonzero(np.asarray(timings) > 500)),
                             per_regime=strata)
    return results


def finalize():
    if (STUDY / 'summary.json').exists():
        raise FileExistsError('Preserve completed analysis')
    directory, protocol = read_protocol(STUDY)
    spec, rows = protocol['spec'], []
    for shard in range(spec['shards']):
        out = directory / f'shard_{shard}'
        complete = json.loads((out / 'complete.json').read_text())
        if complete['protocol_sha256'] != digest(STUDY / 'protocol.json') or complete['episodes_sha256'] != digest(out / 'episodes.json'):
            raise ValueError('Completed shard identity changed')
        part = json.loads((out / 'episodes.json').read_text())
        keys = [(r['initial_index'], r['disturbance'], r['controller']) for r in part]
        if keys != schedule(spec, shard) or len(part) != complete['actual_sumo_runs']:
            raise ValueError('Wrong shard schedule')
        for row in part:
            if row['trace_saved']:
                path = out / f"trace_{row['initial_index']}_{row['disturbance']}_{row['controller']}.json"
                trace_audit(json.loads(path.read_text()), row)
        rows.extend(part)
    if len(rows) != spec['expected_actual_sumo_runs']:
        raise ValueError('Incomplete study')
    results = summarize(rows, spec)
    rows.sort(key=lambda r: (r['initial_index'], r['disturbance'], r['controller']))
    save(STUDY / 'episodes.json', rows)
    report = dict(purpose=spec['purpose'], data_role='development', independent_initials=spec['n_initials'],
                  actual_sumo_runs=len(rows), results=results, risk_certified=False,
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
