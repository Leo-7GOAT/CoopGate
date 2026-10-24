"""Fixed order-balanced SUMO equivalence and complete callback timing study."""
import argparse
import json
import os
from pathlib import Path
from types import FunctionType

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import EpisodeStore, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.memoized_recovery_v37 import MemoizedRecoveryGate
from coopgate.zero_score_projection import FastProjectionGate
from coopgate.recovery_gate_audit_v35 import review_joint_trace
from audit_buffered_scope_v25 import check_calls
import execute_ood_validation_v39 as io
import execute_fair_failure_v44 as frozen_reference
import freeze_fair_comparison_v39 as design
import run_joint_risk_v42 as runtime

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('src/coopgate/zero_score_projection.py', 'tests/test_zero_score_projection.py',
           'tools/benchmark_zero_score_projection.py', 'docs/zero_score_projection_protocol.md')
IMPLEMENTATIONS = dict(memoized=MemoizedRecoveryGate, zero_score=FastProjectionGate)
RULES = ('nominal_performance', 'bank_performance')


def schedule(index):
    names = list(IMPLEMENTATIONS)
    return [[index, repeat, name] for repeat in (0, 1)
            for name in (names if (index+repeat) % 2 == 0 else names[::-1])]


def cases():
    result = []
    for model in ('1802', '1812', '1822'):
        for profile in ('nominal', 'gain_drag', 'lag_mild', 'lag_strong'):
            for rule in RULES:
                index = len(result)
                result.append(dict(index=index, model=model, profile=profile, rule=rule,
                                   schedule=schedule(index)))
    return result


def non_timing(record):
    return dict(row={k: v for k, v in record['row'].items() if k not in ('setup_ms', 'controller_ms')},
                trace=[{k: v for k, v in step.items() if k != 'gate_ms'} for step in record['trace']],
                errors=[call['error'] for call in record['calls']])


def require_equal(records):
    if len(records) != 4 or any(record != records[0] for record in records[1:]):
        raise ValueError('All four executions must preserve every original non-timing field')


def run_episode(case, rule, implementation, model, env):
    def factory(m, p, unused):
        return IMPLEMENTATIONS[implementation](m, p, rule)
    function = FunctionType(runtime.execute_episode.__code__,
        {**runtime.execute_episode.__globals__, 'MemoizedRecoveryGate': factory},
        argdefs=runtime.execute_episode.__defaults__)
    result = function(case, case['policy_id'], model, env)
    result['row']['rule'] = rule
    return result


def summarize(calls):
    result = {}
    for repeat in (None, 0, 1):
        group = {}
        for rule in RULES:
            methods = {}
            for name in IMPLEMENTATIONS:
                values = [c['wall_ms'] for c in calls if c['rule'] == rule and c['implementation'] == name
                          and (repeat is None or c['repeat'] == repeat)]
                if not values or not all(np.isfinite(v) and v >= 0 for v in values):
                    raise ValueError('Missing or invalid complete callback measurements')
                methods[name] = dict(calls=len(values), total_ms=sum(values), mean_ms=float(np.mean(values)),
                    p95_ms=float(np.quantile(values, .95)), max_ms=max(values), over_500ms=sum(v > 500 for v in values))
            old, new = methods['memoized'], methods['zero_score']
            if old['calls'] != new['calls'] or old['total_ms'] <= 0:
                raise ValueError('Unpaired or invalid timing calls')
            methods['total_time_reduction_percent'] = 100*(1-new['total_ms']/old['total_ms'])
            group[rule] = methods
        result['combined' if repeat is None else f'round_{repeat}'] = group
    return result


def verify_inputs(hashes, models, binding_sha, runtime_identity):
    for name, expected in hashes.items():
        if io.digest(ROOT/name) != expected:
            raise ValueError('Bound source changed: '+name)
    for model in models.values():
        if io.digest(ROOT/model['path']) != model['sha256']:
            raise ValueError('Frozen model changed')
    if io.digest(frozen_reference.BINDING) != binding_sha or io.runtime_identity() != runtime_identity:
        raise ValueError('Reference binding or runtime changed')


def run(output):
    for name in SOURCES:
        io.committed(ROOT/name)
    _, reference = frozen_reference.bound_inputs()
    hashes = {**reference['source_sha256'], **{name: io.digest(ROOT/name) for name in SOURCES}}
    models_meta = design.load_models(ROOT)
    binding_sha, identity = io.digest(frozen_reference.BINDING), io.runtime_identity()
    verify_inputs(hashes, models_meta, binding_sha, identity)
    items = cases()
    output.mkdir(parents=True, exist_ok=False)
    io.exclusive_json(output/'plan.json', dict(cases=items, seed=139891001, scenario='nominal', disturbance='clean',
        actual_sumo_runs=96, paired_settings=24, rounds=2, formal_independent_initials=0,
        models=models_meta, source_sha256=hashes, reference_binding_sha256=binding_sha, runtime=identity,
        native_pid=os.getpid(), started_utc=io.utc(),
        timing_scope='Balanced order within each setting; shared desktop; no hard realtime or independent timing inference'))
    plan_sha = io.digest(output/'plan.json')
    torch.set_num_threads(1)
    models = {key: load_coupled(ROOT/value['path'])[0] for key, value in models_meta.items()}
    warm = np.array([[-140., 12.], [-220., 10.], [-190., 12.]])
    for item in items:
        for cls in IMPLEMENTATIONS.values():
            cls(models[item['model']], item['profile'], item['rule']).select(warm, np.zeros(3))
    entries = []
    for item in items:
        case = dict(cluster_id=f"zero-score-dev-{item['index']:03d}", seed=139891001,
                    scenario='nominal', plant=item['profile'], disturbance='clean', policy_id=item['model'])
        directory = output/f"{item['index']:03d}"
        directory.mkdir()
        metadata = dict(plan_sha256=plan_sha, block_index=item['index'])
        store = EpisodeStore(directory/'observations.sqlite', metadata)
        outputs, reviews = [], []
        try:
            with MatchedShiftedMergeEnv(plant=item['profile']) as env:
                for key in item['schedule']:
                    _, repeat, implementation = key
                    result = run_episode(case, item['rule'], implementation, models[item['model']], env)
                    store.append(key, {k: v for k, v in result.items() if k != 'trace'}, result['trace'])
                    check_calls(result['row'], result['trace'], result['calls'])
                    reviewed = review_joint_trace(result['row'], result['trace'], models[item['model']], replay=True)
                    reviews.append(dict(repeat=repeat, implementation=implementation, **reviewed))
                    outputs.append(non_timing(result))
            require_equal(outputs)
            store.seal(item['schedule'])
        finally:
            store.close()
        io.exclusive_json(directory/'complete.json', dict(**metadata, reviews=reviews, all_four_exact=True,
            actual_sumo_runs=4, database_sha256=io.digest(directory/'observations.sqlite'),
            seal_sha256=io.digest(directory/'observations.sqlite.seal.json')))
        entries.append(dict(index=item['index'], files_sha256={n: io.digest(directory/n) for n in
            ('observations.sqlite', 'observations.sqlite.seal.json', 'complete.json')}))
        print(json.dumps(dict(completed_settings=len(entries), expected_settings=24)), flush=True)
    calls, outcomes = [], []
    for item, entry in zip(items, entries):
        directory = output/f"{item['index']:03d}"
        for name, expected in entry['files_sha256'].items():
            if io.digest(directory/name) != expected:
                raise ValueError('Persisted observation changed')
        paired = []
        metadata = dict(plan_sha256=plan_sha, block_index=item['index'])
        for key, body, trace in observations(directory/'observations.sqlite', metadata, item['schedule']):
            paired.append(non_timing(dict(**body, trace=trace)))
            outcomes.append(dict(key=list(key), row=body['row']))
            calls.extend(dict(case=item['index'], rule=item['rule'], repeat=key[1], implementation=key[2],
                              **call) for call in body['calls'])
        require_equal(paired)
    if len(outcomes) != 96 or len(entries) != 24:
        raise ValueError('Incomplete fixed schedule')
    timing = summarize(calls)
    verify_inputs(hashes, models_meta, binding_sha, identity)
    if io.digest(output/'plan.json') != plan_sha:
        raise ValueError('Development plan changed')
    io.exclusive_json(output/'report.json', dict(plan_sha256=plan_sha, actual_sumo_runs=96, paired_settings=24,
        all_original_non_timing_exact=True, new_risk_certificate=False, formal_independent_initials=0,
        timing=timing, outcomes=outcomes, blocks=entries))
    io.exclusive_json(output/'complete.json', dict(plan_sha256=plan_sha, report_sha256=io.digest(output/'report.json'),
        actual_sumo_runs=96, paired_settings=24, native_pid=os.getpid(), finished_utc=io.utc()))
    print(json.dumps(dict(timing=timing)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
