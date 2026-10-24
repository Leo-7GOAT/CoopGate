"""Development collector for the five-way paired mechanism ablation.

No formal launch or outcome-based selection. New formal artifacts require a
separate plan and terminal supervisor; the existing fair/risk plans stay fixed.
"""
from pathlib import Path
from types import FunctionType
from coopgate.candidate_ablation_v43 import CandidateAblationGate, VARIANTS, rule_for
from coopgate.episode_store_v21 import EpisodeStore
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.recovery_gate_audit_v35 import review_joint_trace
from audit_buffered_scope_v25 import check_calls
import run_joint_risk_v42 as risk_runtime
import execute_ood_validation_v39 as files


def execute_episode(case, variant, model, env):
    # Reuse the tested finite-cost/empty-abort accounting. Its factory argument
    # is rebound privately; its module globals and frozen source stay untouched.
    def gate(model, profile, unused_rule):
        return CandidateAblationGate(model, profile, variant)
    run = FunctionType(risk_runtime.execute_episode.__code__,
        {**risk_runtime.execute_episode.__globals__, 'MemoizedRecoveryGate': gate},
        argdefs=risk_runtime.execute_episode.__defaults__)
    method = 'baseline' if variant == 'baseline' else case['policy_id']
    record = run(case, method, None if variant == 'baseline' else model, env)
    record['variant'] = variant
    record['row']['rule'] = rule_for(variant)
    return record


def audit_episode(case, record, model):
    variant, row, trace = record['variant'], record['row'], record['trace']
    if (record['cluster_id'] != case['cluster_id'] or variant not in VARIANTS
            or record['method'] != ('baseline' if variant == 'baseline' else case['policy_id'])
            or row['rule'] != rule_for(variant) or row['seed'] != case['seed']
            or row['scenario'] != case['scenario'] or row['disturbance'] != case['disturbance']
            or row['plant_name'] != case['plant']
            or row['model'] != ('none' if variant == 'baseline' else case['policy_id'])):
        raise ValueError('Ablation identity differs')
    if any(t['candidate_ablation_variant'] != variant for t in trace):
        raise ValueError('Trace candidate mechanism changed')
    shared = risk_runtime.design.shared
    if (shared.array_sha256(row['initial'], decimals=8) != case['initial_sha256']
            or shared.canonical_sha256(row['plant']) != case['dynamics_sha256']):
        raise ValueError('Physical pairing identity differs')
    tape = risk_runtime.disturbance_tape(case['seed'], case['disturbance'], .1, 250)
    if shared.array_sha256(tape) != case['disturbance_sha256']:
        raise ValueError('Disturbance pairing identity differs')
    check_calls(row, trace, record['calls'])
    # The existing auditor reconstructs the physical recurrence independently.
    # Only its replay-controller factory is rebound to the declared variant.
    def gate(model, profile, rule):
        if rule != rule_for(variant):
            raise ValueError('Unexpected replay rule')
        return CandidateAblationGate(model, profile, variant)
    review = FunctionType(review_joint_trace.__code__,
        {**review_joint_trace.__globals__, 'RecoveryJointGate': gate},
        argdefs=review_joint_trace.__defaults__)
    return review(row, trace, None if variant == 'baseline' else model, replay=True)


def collect_case(case, directory, model):
    order = case['variants']
    if len(order) != 5 or set(order) != set(VARIANTS):
        raise ValueError('All five factorial methods required exactly once')
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    metadata = dict(cluster_id=case['cluster_id'], scope='development-only-mechanism-check')
    store = EpisodeStore(directory/'observations.sqlite', metadata)
    reviews = []
    try:
        with MatchedShiftedMergeEnv(plant=case['plant']) as env:
            for variant in order:
                record = execute_episode(case, variant, model, env)
                store.append([case['cluster_id'], variant],
                    {k: v for k, v in record.items() if k != 'trace'}, record['trace'])
                reviews.append(dict(variant=variant, **audit_episode(case, record, model)))
        store.seal([[case['cluster_id'], v] for v in order])
    finally:
        store.close()
    result = dict(**metadata, actual_sumo_runs=5, reviews=reviews,
        database_sha256=files.digest(directory/'observations.sqlite'),
        seal_sha256=files.digest(directory/'observations.sqlite.seal.json'),
        formal_efficacy_samples=0)
    files.exclusive_json(directory/'complete.json', result)
    return result
