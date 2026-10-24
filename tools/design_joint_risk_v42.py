"""Outcome-free schedule for the unchanged V39 joint recovery certificate.

This module does not launch a simulator. A separately bound executor must prove
the previous statistical attempt terminal before starting this attempt.
"""
from collections import Counter
import numpy as np

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import matched_plant_parameters
from coopgate.recovery_risk_v39 import (ATTEMPT, ATTEMPT_ALPHA, CALIBRATION_N,
    CONFIRMATION_N, MODELS, PLANTS, PLANT_COUNT, ADOPTION_MIN, GAIN_MIN_PERCENT, RISK_LIMIT)
import freeze_fair_comparison_v39 as shared
import execute_ood_validation_v39 as files

ROOT = shared.ROOT
OUT = ROOT/'artifacts/v42/joint_risk'
STAGES = ('calibration', 'confirmation')
SEEDS = {'calibration': 842_000_001, 'confirmation': 843_000_001}
STREAMS = {'calibration': 420_101, 'confirmation': 420_201}
COUNTS = {'calibration': CALIBRATION_N, 'confirmation': CONFIRMATION_N}
METHODS = ('baseline', *(str(model) for model in MODELS))
NEW_SOURCES = ('tools/design_joint_risk_v42.py', 'tools/run_joint_risk_v42.py',
    'src/coopgate/joint_risk_data_v42.py', 'tests/test_joint_risk_v42.py',
    'docs/v42_joint_risk_execution.md')


def stage_cases(stage):
    """IID within calibration strata; IID full mixture for confirmation."""
    if stage not in STAGES:
        raise ValueError('Unknown frozen stage')
    count = COUNTS[stage]
    rng = np.random.default_rng(np.random.SeedSequence([STREAMS[stage], count]))
    scenes = rng.integers(len(shared.SCENARIOS), size=count)
    plants = (np.arange(count) % len(PLANTS) if stage == 'calibration'
              else rng.integers(len(PLANTS), size=count))
    modes = rng.integers(len(shared.DISTURBANCES), size=count)
    cases = []
    for index in range(count):
        seed = SEEDS[stage]+index
        scenario, plant, mode = (shared.SCENARIOS[int(scenes[index])],
                                PLANTS[int(plants[index])], shared.DISTURBANCES[int(modes[index])])
        offset = index % len(METHODS)
        order = METHODS[offset:]+METHODS[:offset]
        if index % 2:
            order = order[::-1]
        cases.append(dict(stage=stage, index=index, cluster_id=f'v42:{stage}:{index:04d}',
            seed=seed, scenario=scenario, plant=plant, disturbance=mode, methods=list(order),
            initial_sha256=shared.array_sha256(shared.planned_initial(seed, scenario), decimals=8),
            dynamics_sha256=shared.canonical_sha256(matched_plant_parameters(seed, plant, .1)),
            disturbance_sha256=shared.array_sha256(disturbance_tape(seed, mode, .1, 250))))
    return cases


def contract():
    return dict(attempt=ATTEMPT, alpha=ATTEMPT_ALPHA, models=list(MODELS),
        risk_limit=RISK_LIMIT, adoption_lower_minimum=ADOPTION_MIN,
        paired_mean_gain_lower_strict_minimum_percent=GAIN_MIN_PERCENT,
        calibration=dict(initials=CALIBRATION_N, iid_within_plant=True,
            fixed_plant_counts={name: PLANT_COUNT for name in PLANTS}, weights=[.25]*4),
        confirmation=dict(initials=CONFIRMATION_N, iid_full_mixture=True,
            balancing_or_rejection_permitted=False),
        baseline='MemoizedRecoveryGate(None, declared_class, baseline)',
        candidate='MemoizedRecoveryGate(fixed_model, declared_class, bank_performance)',
        replay_reference='RecoveryJointGate with same rule/class/model and full executed history',
        performance_gain_inference='paired t: normal-IID exact, otherwise asymptotic finite variance',
        conditional_risk_inference='V39 stratum joint-harm/adoption bounds and strong-lag exact binomial',
        gate_thresholds_changed=False, calibration_selects_controller=False,
        previous_attempt_terminal_required=True,
        complete_calibration_before_confirmation=True,
        confirmation_runs_even_if_calibration_certificate_fails=True,
        zero_observed_baseline_and_candidate_failures_required=True,
        failure='Incomplete horizon, abort, collision, route violation; no callback deadline cutoff',
        timing='Measured callback latency, >500ms diagnostic only; no hard-real-time claim',
        harm='Candidate failure OR baseline failure OR any cost/arrival regression >1e-9',
        baseline_failure_handling='Retain capped cost, flag relative comparison unavailable; impute harm conservatively; reject deployment',
        zero_adoption='Require same cost/arrival when both executions succeed; retain failures before first adoption',
        selection='No rejected initial, sample replacement, model selection, threshold tuning or partial effect analysis',
        new_formal_initials=CALIBRATION_N+CONFIRMATION_N,
        expected_sumo_runs=4*(CALIBRATION_N+CONFIRMATION_N))


def build_plan():
    sources = set(NEW_SOURCES) | set(shared.SOURCE_FILES)
    sources.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'src/coopgate').glob('*.py'))
    return dict(schema='coopgate.joint-risk.v42', contract=contract(),
        models=shared.load_models(ROOT), stages={stage: stage_cases(stage) for stage in STAGES},
        source_sha256={name: files.digest(ROOT/name) for name in sorted(sources)},
        model_plan_sha256=files.digest(ROOT/shared.MODEL_PLAN),
        new_study_started=False, previous_attempt_effects_read=False)


def validate_plan(plan, *, verify_bytes=True):
    if plan['schema'] != 'coopgate.joint-risk.v42' or plan['contract'] != contract():
        raise ValueError('Joint claim, controller or acceptance contract changed')
    if tuple(plan['stages']) != STAGES or any(plan['stages'][s] != stage_cases(s) for s in STAGES):
        raise ValueError('Exact predeclared independent schedule changed')
    if Counter(c['plant'] for c in plan['stages']['calibration']) != {p: PLANT_COUNT for p in PLANTS}:
        raise ValueError('Calibration strata changed')
    if not set(NEW_SOURCES) <= set(plan['source_sha256']):
        raise ValueError('Missing joint risk dependency')
    if verify_bytes:
        if (plan['models'] != shared.load_models(ROOT)
                or plan['model_plan_sha256'] != files.digest(ROOT/shared.MODEL_PLAN)):
            raise ValueError('Fixed model family changed')
        for name, expected in plan['source_sha256'].items():
            if files.digest(ROOT/name) != expected:
                raise ValueError('Joint risk source changed: '+name)
    return plan
