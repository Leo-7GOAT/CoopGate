"""Parallel independent V42 audits; preserve the original finalizer and inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

import torch

from coopgate.episode_store_v21 import encode, observations
import execute_joint_risk_v42 as original
import finalize_episode_keys as keys

ROOT = original.ROOT
OUT = ROOT/'artifacts/joint_risk_parallel_replay_20261027'
PLAN = OUT/'plan.json'
SCRIPT = 'tools/parallel_joint_risk_replay_20261027.py'
SOURCES = (SCRIPT, 'tools/execute_joint_risk_recovered_20261027.py',
           'tests/test_parallel_joint_risk_replay_20261027.py',
           'operations/benchmark_parallel_joint_replay_20261027.py',
           'docs/parallel_joint_risk_replay_20261027.md')
WORKERS = 3
EXECUTION_BINDING = original.OUT/'parallel_replay_binding.json'
_MODELS = None


def model_signature(model):
    if model is None:
        return 'baseline:none'
    result = hashlib.sha256(encode(dict(training=model.training)))
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(encode([name, str(array.dtype), list(array.shape)]))
        result.update(array.tobytes())
    return result.hexdigest()


def argument_key(case, record, model):
    return hashlib.sha256(encode([case, record, model_signature(model)])).hexdigest()


def initialize(models):
    global _MODELS
    torch.set_num_threads(1)
    _MODELS = original.load_models(dict(models=models))


def audit_block(job):
    """Each process loads its own pinned models and calls the unchanged auditor."""
    directory = Path(job['directory'])
    names = ('observations.sqlite', 'observations.sqlite.seal.json', 'complete.json')
    before = {name:original.io.digest(directory/name) for name in names}
    result = []
    case = job['case']
    for key, body, trace in observations(directory/'observations.sqlite', job['metadata'],
                                         original.collect.schedule(case)):
        if list(key) != [case['cluster_id'], body['method']]:
            raise ValueError('Parallel replay key differs from the frozen schedule')
        model = None if body['method'] == 'baseline' else _MODELS[body['method']]
        record = dict(**body, trace=trace)
        review = original.collect.audit_record(case, record, model)
        result.append(dict(argument_sha256=argument_key(case,record,model), review=review))
    if len(result) != 4 or any(original.io.digest(directory/n) != h for n,h in before.items()):
        raise ValueError('Incomplete or changed parallel replay block')
    value = dict(cluster_id=case['cluster_id'], source_hashes=before, reviews=result)
    if job.get('result_path'):
        path = Path(job['result_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        original.io.exclusive_json(path,value)
    return value


def replay_blocks(jobs, models, workers=WORKERS):
    if workers != WORKERS:
        raise ValueError('Only the registered worker count is allowed')
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                             initargs=(models,)) as pool:
        # map preserves the original stage/case order. All workers join before
        # the original quiet-project finalizer is entered.
        return list(pool.map(audit_block, jobs, chunksize=1))


class AuditLookup:
    def __init__(self, blocks):
        self.pending = {}
        self.used = set()
        for block in blocks:
            for item in block['reviews']:
                key = item['argument_sha256']
                if key in self.pending:
                    raise ValueError('Duplicate parallel replay argument identity')
                self.pending[key] = item['review']
        self.count = len(self.pending)

    def review(self, case, record, model):
        key = argument_key(case,record,model)
        if key not in self.pending or key in self.used:
            raise ValueError('Original finalizer arguments differ or were replayed twice')
        self.used.add(key)
        return dict(self.pending[key])

    def require_complete(self):
        if self.used != set(self.pending):
            raise ValueError('Every original scheduled audit must be consumed before inference')


def adapted_finalizer(function, lookup, before_inference=None):
    namespace = dict(function.__globals__)
    collect = namespace['collect']
    proxy = SimpleNamespace(**vars(collect))
    proxy.audit_record = lookup.review
    infer = namespace['certificate_from_complete_records']
    def checked_inference(*args, **kwargs):
        lookup.require_complete()
        if before_inference is not None:
            before_inference()
        return infer(*args, **kwargs)
    namespace.update(collect=proxy, certificate_from_complete_records=checked_inference)
    result = FunctionType(function.__code__,namespace,function.__name__,
                          function.__defaults__,function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return keys.compatible_function(result)


def contract_inputs():
    original.io.committed(PLAN)
    plan = original.io.read(PLAN)
    if (plan['workers'] != WORKERS or plan['joint_plan_sha256'] != original.io.digest(original.PLAN)
            or plan['scientific_functions_changed'] or plan['statistical_sample_size_changed']):
        raise ValueError('Parallel replay registration differs')
    for name, expected in plan['source_sha256'].items():
        original.io.committed(ROOT/name)
        if original.io.digest(ROOT/name) != expected:
            raise ValueError('Registered replay source changed: '+name)
    return plan


def bind_registration():
    contract_inputs()
    binding = original.io.read(original.BINDING)
    for name in (SCRIPT,'tools/execute_joint_risk_recovered_20261027.py'):
        if binding['sources'].get(name) != original.io.digest(ROOT/name):
            raise ValueError('Parallel replay execution sources were not frozen by joint binding')
    original.io.exclusive_json(EXECUTION_BINDING,dict(
        registration_sha256=original.io.digest(PLAN),
        joint_binding_sha256=original.io.digest(original.BINDING),
        frozen_before_any_affected_formal_collection=True))


def bound_registration_inputs():
    plan = contract_inputs()
    original.io.committed(EXECUTION_BINDING)
    original.io.committed(original.BINDING)
    bound = original.io.read(EXECUTION_BINDING)
    if (bound['registration_sha256'] != original.io.digest(PLAN)
            or bound['joint_binding_sha256'] != original.io.digest(original.BINDING)
            or not bound['frozen_before_any_affected_formal_collection']):
        raise ValueError('Parallel replay registration or joint binding changed')
    return plan


def finalize(namespace):
    registered = bound_registration_inputs()
    plan, binding = namespace['bound_inputs']()
    namespace['require_quiet'](original.io.native_snapshot())
    plan_sha, binding_sha = original.io.digest(original.PLAN), original.io.digest(original.BINDING)
    for stage in original.design.STAGES:
        namespace['stage_terminal'](stage,plan_sha,binding_sha)
    target = original.OUT/'parallel_replay'
    if target.exists() or (original.OUT/'analysis').exists():
        raise FileExistsError('Preserve an existing replay prefix; no implicit resume')
    target.mkdir()
    jobs = []
    for stage, cases in plan['stages'].items():
        for case in cases:
            jobs.append(dict(case=case,
                directory=str(original.OUT/stage/'blocks'/f"{case['index']:05d}"),
                metadata=dict(stage=stage,cluster_id=case['cluster_id'],
                              plan_sha256=plan_sha,binding_sha256=binding_sha),
                result_path=str(target/stage/f"{case['index']:05d}.json")))
    started = time.perf_counter()
    blocks = replay_blocks(jobs,plan['models'])
    lookup = AuditLookup(blocks)
    if lookup.count != 16384:
        raise ValueError('Full 16384-episode independent replay required')
    if bound_registration_inputs() != registered:
        raise ValueError('Replay registration changed during execution')
    def still_bound():
        if bound_registration_inputs() != registered:
            raise ValueError('Replay registration changed before inference')
    adapted_finalizer(namespace['finalize'],lookup,before_inference=still_bound)()
    lookup.require_complete()
    original.io.exclusive_json(original.OUT/'analysis/parallel_replay_execution.json',dict(
        registration_sha256=original.io.digest(PLAN), workers=WORKERS,
        replay_binding_sha256=original.io.digest(EXECUTION_BINDING),
        audited_episodes=lookup.count, blocks=len(blocks),
        every_original_argument_matched=True, every_audit_consumed_before_inference=True,
        original_finalizer_code_unchanged=True, original_auditor_function_unchanged=True,
        original_module_globals_mutated=False, original_terminal_and_quiet_checks_retained=True,
        new_sumo_runs=0, statistical_sample_size_changed=False,
        elapsed_shared_host_seconds=time.perf_counter()-started,
        standalone_latency_or_safety_claim=False))


def freeze():
    if PLAN.exists() or original.BINDING.exists() or any(
            (original.OUT/stage/'execution').exists() for stage in original.design.STAGES):
        raise FileExistsError('Register before any affected formal binding or execution')
    benchmark = OUT/'benchmark/report.json'
    proof = original.io.read(benchmark)
    if (not proof['all_original_audit_results_exact'] or proof['new_sumo_runs'] != 0
            or proof['source_sha256'][SCRIPT] != original.io.digest(ROOT/SCRIPT)):
        raise ValueError('Exact original-data audit benchmark required')
    for name in SOURCES:
        original.io.committed(ROOT/name)
    original.io.committed(benchmark)
    OUT.mkdir(parents=True,exist_ok=True)
    original.io.exclusive_json(PLAN,dict(workers=WORKERS,
        joint_plan_sha256=original.io.digest(original.PLAN),
        source_sha256={name:original.io.digest(ROOT/name) for name in SOURCES},
        benchmark_sha256=original.io.digest(benchmark),
        scientific_functions_changed=False, statistical_sample_size_changed=False,
        numerical_inference_changed=False, formal_efficacy_outcomes_read=False,
        complete_native_exits_before_parallel_replay=True,
        original_finalizer_code_unchanged=True, original_auditor_function_unchanged=True,
        no_replay_resume_or_partial_inference=True))
    print('Registered three-worker full replay; no simulation started',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze',))
    globals()[parser.parse_args().command]()
