"""Freeze/check the V39 IID fair-comparison plan before any SUMO execution.

The builder reads only dependency identities and the structural V32 model plan.
It never reads an episode, trace, analysis report, or running-study outcome.
There is intentionally no simulator-launch command in this file.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.fair_comparison_v39 import (
    MASKS, METHODS, CaseIdentity, ComputeBudget, DecisionEvidence,
    FairComparisonProtocol, ProtocolSpec, fair_schedule, fingerprint,
)
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.shifted_sumo_v19 import PLANTS, SCENARIOS, shifted_initial
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
from execute_prospective_v23 import exclusive_json, runtime_identity, sources_committed, utc
from profile_phase_runtime_v24 import committed, digest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v39/fair_comparison_preregistered'
SCRIPT = 'tools/freeze_fair_comparison_v39.py'
TEST = 'tests/test_fair_comparison_plan_v39.py'
CORE = 'src/coopgate/fair_comparison_v39.py'
CORE_TEST = 'tests/test_fair_comparison_v39.py'
DOC = 'docs/v39_fair_comparison_protocol.md'
MODEL_PLAN = 'artifacts/v32/online_actuator_preflight/plan.json'
CALIBRATION = 'artifacts/v13/ego/calibration/calibration.json'
CALIBRATION_PROTOCOL = 'artifacts/v13/ego/calibration/protocol.json'
MODELS = ('1802', '1812', '1822')
CALIBRATION_KEYS = {'1802':'701', '1812':'711', '1822':'721'}
DISTURBANCES = ('clean', 'uniform', 'bias', 'pulse')
PRIMARY_CLUSTERS = 640
CLUSTERS_PER_BLOCK = 4
SEED_BASE = 739_000_001
MIXTURE_STREAM = 390_240_101
MAX_DECISIONS = 50

BASELINE_COMPONENT_FILES = (
    'src/coopgate/actuator_recovery_v34.py',
    'src/coopgate/buffered_baseline_v25.py',
    'src/coopgate/fast_v8.py',
    'src/coopgate/marl_v12.py',
    'src/coopgate/merge_v4.py',
    'src/coopgate/robust_v9.py',
    'src/coopgate/single_v12.py',
)
PROPOSED_GATE_COMPONENT_FILES = (
    'src/coopgate/recovery_joint_gate_v35.py',
    'src/coopgate/recovery_forecast_v35.py',
    'src/coopgate/actuator_prediction_roundoff_v31.py',
    'src/coopgate/actuator_recovery_v34.py',
    'src/coopgate/arrival_phase_v20.py',
    'src/coopgate/buffered_baseline_v25.py',
    'src/coopgate/cooperative_v15.py',
    'src/coopgate/ego_v13.py',
    'src/coopgate/fast_v8.py',
    'src/coopgate/joint_planner_v15.py',
    'src/coopgate/marl_v12.py',
    'src/coopgate/merge_v4.py',
    'src/coopgate/online_actuator_gate_v32.py',
    'src/coopgate/physical_switch_v27.py',
    'src/coopgate/portfolio_gate_v19.py',
    'src/coopgate/repair_portfolio_v20.py',
    'src/coopgate/robust_v9.py',
    'src/coopgate/single_v12.py',
    'src/coopgate/stress_prediction_v17.py',
    'src/coopgate/temporal_portfolio_v20.py',
)
CONFIDENCE_COMPONENT_FILES = (
    'src/coopgate/confidence_v12.py',
    'src/coopgate/shield.py',
    CORE,
)
ORACLE_COMPONENT_FILES = (
    'src/coopgate/recovery_joint_gate_v35.py',
    'src/coopgate/physical_switch_v27.py',
    'src/coopgate/actuator_recovery_v34.py',
    'src/coopgate/robust_v9.py',
)

SOURCE_FILES = (
    CORE, SCRIPT, TEST, CORE_TEST, DOC,
    *BASELINE_COMPONENT_FILES,
    *PROPOSED_GATE_COMPONENT_FILES,
    *CONFIDENCE_COMPONENT_FILES,
    *ORACLE_COMPONENT_FILES,
    'src/coopgate/matched_shift_v21.py',
    'src/coopgate/shifted_sumo_v19.py',
    'src/coopgate/disturbance_v9.py',
)
SOURCE_FILES = tuple(dict.fromkeys(SOURCE_FILES))
DEPENDENCY_FILES = SOURCE_FILES + (MODEL_PLAN, CALIBRATION, CALIBRATION_PROTOCOL)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_sha256(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def array_sha256(values, *, decimals=None):
    array = np.asarray(values, dtype='<f8')
    if decimals is not None:
        array = np.round(array, decimals=decimals)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order='C')).hexdigest()


def forecast_bank_sha256():
    names, tapes = temporal_stress_bank('sampled')
    if len(names) != 33 or np.asarray(tapes).shape != (33, 250, 3):
        raise ValueError('The actual V35 33-path forecast bank changed')
    # RecoveryJointGate scales the legacy +/-0.5 bank to its declared +/-0.1
    # external-innovation radius before any forecast.  Hash executed bytes.
    tapes = np.ascontiguousarray(np.asarray(tapes, dtype='<f8') * (.1 / .5))
    sha = hashlib.sha256(json.dumps(list(names), separators=(',', ':')).encode('utf-8'))
    sha.update(tapes.tobytes(order='C'))
    return sha.hexdigest()


def file_hashes(root, names=DEPENDENCY_FILES):
    root = Path(root)
    result = {}
    for name in names:
        path = root/name
        if not path.is_file():
            raise FileNotFoundError('Missing frozen fair-comparison dependency: '+name)
        result[name] = digest(path)
    return result


def verify_file_hashes(root, expected):
    actual = file_hashes(root, tuple(expected))
    if actual != expected:
        changed = sorted(name for name in set(actual) | set(expected)
                         if actual.get(name) != expected.get(name))
        raise ValueError('Fair-comparison dependency bytes changed: '+','.join(changed))


def component_sha256(hashes, names):
    selected = {name:hashes[name] for name in names}
    return canonical_sha256(selected)


def planned_initial(seed, scenario):
    if scenario in ('nominal', 'dense'):
        rng = np.random.default_rng(seed)
        velocity = rng.uniform(8., 14., 3) if scenario == 'nominal' else rng.uniform(12., 17., 3)
        position = -rng.uniform(70., 110., 3) if scenario == 'nominal' else -rng.uniform(70., 85., 3)
        position[2] = position[0]-rng.uniform(30., 45.)
        return np.column_stack((position, velocity))
    return shifted_initial(seed, scenario)


def protocol_from_dict(value):
    value = deepcopy(value)
    value['budget'] = ComputeBudget(**value['budget'])
    value['methods'] = tuple(value['methods'])
    return ProtocolSpec(**value)


def case_from_dict(value):
    return CaseIdentity(**value)


def load_models(root, model_plan=MODEL_PLAN):
    value = read_json(Path(root)/model_plan)
    models = value.get('models')
    if not isinstance(models, dict) or tuple(models) != MODELS:
        raise ValueError('The three fixed V35 learned policies are required in frozen order')
    result = {}
    for name in MODELS:
        item = models[name]
        path, expected = item.get('path'), item.get('sha256')
        if not isinstance(path, str) or not isinstance(expected, str) or digest(Path(root)/path) != expected:
            raise ValueError('Frozen learned policy changed: '+name)
        result[name] = dict(path=path, sha256=expected,
            training_seed=item.get('training_seed'), updates=item.get('updates'))
    return result


def iid_cases(models, count=PRIMARY_CLUSTERS):
    if count != PRIMARY_CLUSTERS:
        raise ValueError('The pre-registered primary IID cluster count cannot change')
    rng = np.random.default_rng(np.random.SeedSequence([MIXTURE_STREAM, count]))
    scenario_index = rng.integers(len(SCENARIOS), size=count)
    plant_index = rng.integers(len(PLANTS), size=count)
    disturbance_index = rng.integers(len(DISTURBANCES), size=count)
    model_index = rng.integers(len(MODELS), size=count)
    result = []
    for index in range(count):
        seed = SEED_BASE+index
        scenario = SCENARIOS[int(scenario_index[index])]
        plant = PLANTS[int(plant_index[index])]
        mode = DISTURBANCES[int(disturbance_index[index])]
        model = MODELS[int(model_index[index])]
        initial = planned_initial(seed, scenario)
        dynamics = matched_plant_parameters(seed, plant, .1)
        tape = disturbance_tape(seed, mode, .1, 250)
        cluster = f'v39-iid-{index:04d}'
        result.append(CaseIdentity(case_id=f'{cluster}:{model}', independent_cluster=cluster,
            policy_id=model, policy_sha256=models[model]['sha256'], scenario=scenario,
            initial_id=seed, dynamics=plant, disturbance=mode,
            initial_sha256=array_sha256(initial, decimals=8),
            dynamics_sha256=canonical_sha256(dynamics),
            disturbance_sha256=array_sha256(tape)))
    return tuple(result)


def grouped_schedule(cases):
    schedule = fair_schedule(cases)
    by_case = {}
    for assignment in schedule:
        by_case.setdefault(assignment.case.case_id, []).append(assignment)
    groups = []
    for block_index, start in enumerate(range(0, len(cases), CLUSTERS_PER_BLOCK)):
        selected = cases[start:start+CLUSTERS_PER_BLOCK]
        assignments = []
        for case in selected:
            assignments.extend(dict(case_id=case.case_id, method=item.method, order=item.order)
                               for item in by_case[case.case_id])
        groups.append(dict(block_index=block_index,
            case_ids=[case.case_id for case in selected], assignments=assignments,
            actual_sumo_runs=len(assignments)))
    return groups


def build_plan(root=ROOT, *, created_utc=None, runtime=None):
    root = Path(root)
    dependencies = file_hashes(root)
    models = load_models(root)
    bank_sha = forecast_bank_sha256()
    protocol = ProtocolSpec(
        baseline_implementation_sha256=component_sha256(dependencies,
            BASELINE_COMPONENT_FILES),
        proposed_gate_implementation_sha256=component_sha256(dependencies,
            PROPOSED_GATE_COMPONENT_FILES),
        calibration_sha256=dependencies[CALIBRATION],
        confidence_score_definition_sha256=component_sha256(dependencies,
            CONFIDENCE_COMPONENT_FILES),
        forecast_bank_sha256=bank_sha,
        physical_oracle_sha256=component_sha256(dependencies,
            ORACLE_COMPONENT_FILES),
        budget=ComputeBudget(decision_timeout_ms=500.))
    cases = iid_cases(models)
    groups = grouped_schedule(cases)
    episodes = len(cases)*len(METHODS)
    maximum_decisions = episodes*MAX_DECISIONS
    plan = dict(version='v39-fair-comparison-preregistered', utc=created_utc or utc(),
        runtime=runtime if runtime is not None else runtime_identity(),
        dependency_sha256=dependencies, model_plan=MODEL_PLAN, models=models,
        calibration=dict(path=CALIBRATION, protocol=CALIBRATION_PROTOCOL,
            historical_model_key_map=CALIBRATION_KEYS,
            decision_inequality='predicted_delta + q <= -minimum_gain',
            adapter_only=True, inherited_risk_certificate=False,
            values_read_during_plan_build=False),
        protocol=asdict(protocol), protocol_sha256=fingerprint(protocol),
        forecast_bank_sha256=bank_sha,
        forecast_bank_contract=dict(kind='sampled', paths=33, horizon_steps=250,
            source_radius=.5, executed_external_innovation_radius=.1, scale=.2,
            hash_covers='ordered names plus executed little-endian float64 tape bytes'),
        methods=list(METHODS), method_count=len(METHODS),
        cases=[asdict(case) for case in cases], independent_clusters=len(cases),
        groups=groups, blocks=len(groups), actual_sumo_runs=episodes,
        design=dict(kind='iid_frozen_uniform_mixture_with_replacement',
            mixture_stream=MIXTURE_STREAM, seed_base=SEED_BASE,
            support=dict(scenarios=list(SCENARIOS), plants=list(PLANTS),
                         disturbances=list(DISTURBANCES), models=list(MODELS)),
            no_rejection_no_cell_completion_no_outcome_selection=True,
            pooled_exact_binomial_target='marginal over the frozen uniform mixture',
            fixed_cell_results='secondary descriptive only; never pooled as IID repetitions'),
        comparison_scope=dict(
            proposed_candidate_set='projected/repaired search over at most 28 commands',
            ordinary_comparator_candidate_set='raw learned/baseline masks only',
            equal_shadow_computation=True,
            pure_gate_criterion_isolation=False,
            limitation=('Candidate sets differ: this study compares complete deployed selectors, '
                        'not only their scalar admission criteria.')),
        run_budget=dict(sumo_episodes=episodes, maximum_decisions_per_episode=MAX_DECISIONS,
            maximum_control_decisions=maximum_decisions,
            fixed_forecast_paths_per_packet=protocol.budget.forecast_paths,
            maximum_proposed_gate_forecast_calls_per_packet=
                protocol.budget.maximum_proposed_gate_forecast_calls,
            maximum_fixed_path_forecasts=maximum_decisions*protocol.budget.forecast_paths,
            maximum_proposed_gate_forecast_calls=
                maximum_decisions*protocol.budget.maximum_proposed_gate_forecast_calls,
            maximum_physical_mask_checks=maximum_decisions*len(MASKS),
            common_decision_timeout_ms=protocol.budget.decision_timeout_ms,
            aggregate_controller_deadline_hours=maximum_decisions*
                protocol.budget.decision_timeout_ms/1000./3600.,
            maximum_simulated_episode_hours=episodes*25./3600.,
            clusters_per_block=CLUSTERS_PER_BLOCK, blocks=len(groups)),
        primary_decision=dict(metric=protocol.primary_metric,
            adoption_lower_minimum=protocol.adoption_minimum,
            paired_cluster_gain_t_lower_percent_minimum=protocol.gain_minimum_percent,
            total_error=protocol.family_error,
            error_shares=dict(risk=protocol.risk_error_share,
                              adoption=protocol.adoption_error_share,
                              gain=protocol.gain_error_share),
            proposed_method='proposed_bank_performance_gate',
            no_eligible_method_result=None, always_baseline_eligible=False),
        fresh_initials=True, prior_seed_namespace_excluded=True,
        existing_or_running_effectiveness_outcomes_read=False,
        source_or_frozen_study_mutated=False, simulator_started=False)
    validate_plan_structure(plan)
    return plan


def validate_plan_structure(plan):
    if (plan.get('version') != 'v39-fair-comparison-preregistered'
            or plan.get('methods') != list(METHODS) or plan.get('method_count') != len(METHODS)
            or plan.get('independent_clusters') != PRIMARY_CLUSTERS
            or plan.get('actual_sumo_runs') != PRIMARY_CLUSTERS*len(METHODS)
            or plan.get('blocks') != len(plan.get('groups', []))
            or plan.get('existing_or_running_effectiveness_outcomes_read') is not False
            or plan.get('simulator_started') is not False):
        raise ValueError('Changed V39 fair-comparison scope or outcome isolation')
    protocol = protocol_from_dict(plan['protocol'])
    if (fingerprint(protocol) != plan['protocol_sha256']
            or protocol.budget.decision_timeout_ms != 500.
            or protocol.methods != METHODS or len(METHODS) != 8):
        raise ValueError('Changed common evidence, method family or 500 ms budget')
    design = plan.get('design', {})
    if (design.get('kind') != 'iid_frozen_uniform_mixture_with_replacement'
            or design.get('no_rejection_no_cell_completion_no_outcome_selection') is not True
            or design.get('pooled_exact_binomial_target') !=
                'marginal over the frozen uniform mixture'):
        raise ValueError('Primary clusters must remain IID draws from the frozen mixture')
    scope = plan.get('comparison_scope', {})
    if (scope.get('proposed_candidate_set') !=
            'projected/repaired search over at most 28 commands'
            or scope.get('ordinary_comparator_candidate_set') !=
                'raw learned/baseline masks only'
            or scope.get('equal_shadow_computation') is not True
            or scope.get('pure_gate_criterion_isolation') is not False):
        raise ValueError('Candidate-set limitation must remain explicit before execution')
    bank = plan.get('forecast_bank_contract', {})
    if bank != dict(kind='sampled', paths=33, horizon_steps=250, source_radius=.5,
            executed_external_innovation_radius=.1, scale=.2,
            hash_covers='ordered names plus executed little-endian float64 tape bytes'):
        raise ValueError('Executed V35 forecast-bank scaling or byte contract changed')
    cases = tuple(case_from_dict(value) for value in plan['cases'])
    if (len(cases) != PRIMARY_CLUSTERS or len({c.case_id for c in cases}) != len(cases)
            or len({c.independent_cluster for c in cases}) != len(cases)
            or len({c.initial_id for c in cases}) != len(cases)
            or [c.initial_id for c in cases] != list(range(SEED_BASE, SEED_BASE+PRIMARY_CLUSTERS))):
        raise ValueError('Fresh independent cluster identities or seed namespace changed')
    models = plan.get('models', {})
    if tuple(models) != MODELS or any(case.policy_sha256 != models[case.policy_id]['sha256'] for case in cases):
        raise ValueError('Case/model identity differs from the three frozen policies')
    expected_cases = iid_cases(models)
    if cases != expected_cases:
        raise ValueError('IID mixture draw, initial, dynamics or disturbance identity changed')
    expected_groups = grouped_schedule(cases)
    if plan['groups'] != expected_groups:
        raise ValueError('Every IID cluster must have the exact eight-method crossed schedule')
    budget = plan.get('run_budget', {})
    maximum_decisions = plan['actual_sumo_runs']*MAX_DECISIONS
    if (budget.get('sumo_episodes') != plan['actual_sumo_runs']
            or budget.get('maximum_control_decisions') != maximum_decisions
            or budget.get('common_decision_timeout_ms') != 500.
            or budget.get('maximum_proposed_gate_forecast_calls') !=
                maximum_decisions*protocol.budget.maximum_proposed_gate_forecast_calls):
        raise ValueError('Declared execution budget differs from the exact crossed schedule')
    primary = plan.get('primary_decision', {})
    if (primary.get('proposed_method') != 'proposed_bank_performance_gate'
            or primary.get('adoption_lower_minimum') != .20
            or primary.get('paired_cluster_gain_t_lower_percent_minimum') != .5
            or primary.get('always_baseline_eligible') is not False):
        raise ValueError('Primary decision or anti-degenerate usefulness constraints changed')
    return plan


def validate_dependencies(plan, root=ROOT):
    root = Path(root)
    verify_file_hashes(root, plan['dependency_sha256'])
    models = load_models(root, plan['model_plan'])
    if models != plan['models']:
        raise ValueError('Model identity differs from the recovery plan')
    if forecast_bank_sha256() != plan['forecast_bank_sha256']:
        raise ValueError('Actual 33-path forecast bank changed')
    protocol = protocol_from_dict(plan['protocol'])
    hashes = plan['dependency_sha256']
    expected = dict(
        baseline_implementation_sha256=component_sha256(hashes,
            BASELINE_COMPONENT_FILES),
        proposed_gate_implementation_sha256=component_sha256(hashes,
            PROPOSED_GATE_COMPONENT_FILES),
        calibration_sha256=hashes[CALIBRATION],
        confidence_score_definition_sha256=component_sha256(hashes,
            CONFIDENCE_COMPONENT_FILES),
        forecast_bank_sha256=plan['forecast_bank_sha256'],
        physical_oracle_sha256=component_sha256(hashes,
            ORACLE_COMPONENT_FILES))
    for name, value in expected.items():
        if getattr(protocol, name) != value:
            raise ValueError('Protocol component hash differs: '+name)
    return plan


def plan_check(plan, root=ROOT):
    validate_plan_structure(plan)
    validate_dependencies(plan, root)
    return plan


def route_block(plan, block_index, evidence_by_case):
    """Pure one-decision executor skeleton; performs no SUMO or file writes."""
    validate_plan_structure(plan)
    if (isinstance(block_index, bool) or not isinstance(block_index, int)
            or not 0 <= block_index < len(plan['groups'])):
        raise ValueError('Unknown fair-comparison block')
    group = plan['groups'][block_index]
    if set(evidence_by_case) != set(group['case_ids']):
        raise ValueError('One common evidence packet per block case is required')
    cases = {value['case_id']:case_from_dict(value) for value in plan['cases']
             if value['case_id'] in group['case_ids']}
    protocol = FairComparisonProtocol(protocol_from_dict(plan['protocol']))
    decisions = []
    evidence_hashes = {}
    for assignment in group['assignments']:
        case_id, method = assignment['case_id'], assignment['method']
        evidence = evidence_by_case[case_id]
        if not isinstance(evidence, DecisionEvidence) or evidence.case != cases[case_id]:
            raise ValueError('Evidence packet differs from its frozen case identity')
        decision = protocol.select(method, evidence)
        prior = evidence_hashes.setdefault(case_id, decision.evidence_sha256)
        if prior != decision.evidence_sha256:
            raise ValueError('Methods in one case did not consume identical evidence bytes')
        decisions.append(dict(case_id=case_id, method=method, order=assignment['order'],
            evidence_sha256=decision.evidence_sha256, decision=asdict(decision)))
    if len(decisions) != group['actual_sumo_runs']:
        raise ValueError('Incomplete one-block routing')
    return decisions


def freeze():
    if OUT.exists():
        raise FileExistsError('Preserve the first V39 fair-comparison freeze')
    plan = build_plan()
    plan_check(plan)
    sources_committed([ROOT/name for name in SOURCE_FILES])
    OUT.mkdir(parents=True, exist_ok=False)
    exclusive_json(OUT/'plan.json', plan)
    print(json.dumps(dict(independent_clusters=plan['independent_clusters'],
        methods=plan['method_count'], actual_sumo_runs=plan['actual_sumo_runs'],
        blocks=plan['blocks'], plan_sha256=digest(OUT/'plan.json'))), flush=True)


def check():
    committed(OUT/'plan.json')
    plan = read_json(OUT/'plan.json')
    plan_check(plan)
    print(json.dumps(dict(plan_sha256=digest(OUT/'plan.json'),
        dependencies=len(plan['dependency_sha256']), methods=plan['method_count'],
        independent_clusters=plan['independent_clusters'], ready_for_separate_executor=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'check'))
    globals()[parser.parse_args().command]()
