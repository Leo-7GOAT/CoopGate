"""Freeze validation rules before outcomes; select only after original exits.

This tool does not launch calibration. The complete execution protocol and
executor must also be frozen before generating its new independent initials.
"""
import argparse
import json
from pathlib import Path
import shutil

from coopgate.prospective_validation_v21 import (
    ADOPTION_MIN, CALIBRATION_N, CONFIRMATION_N, FAMILIES, GAIN_PERCENT_MIN, MODELS, MODES,
    RISK_LIMIT, TOTAL_ATTEMPT_ERROR, attempt_error, profile_names, select_development_family,
    validate_attempt_history,
)
from design_risk_power_v20 import exact_cutoffs, joint_lower_bound, single_target_power
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol
from study_phase_repair_v21 import validate_worker_exits

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / 'artifacts/v21/phase_development'
RULES = ROOT / 'artifacts/v21/prospective_validation_rules'
ATTEMPTS = ROOT / 'artifacts/v21/validation_attempts'
SOURCES = ('src/coopgate/prospective_validation_v21.py', 'src/coopgate/family_risk_v20.py',
    'src/coopgate/comparison_v19.py', 'src/coopgate/risk_comparators_v19.py',
    'src/coopgate/joint_admission_v15.py', 'tools/prepare_prospective_validation_v21.py',
    'tools/design_risk_power_v20.py', 'tools/probe_disturbed_gate_v19.py',
    'tools/study_gate_v19.py', 'tools/study_phase_repair_v21.py', 'tools/start_final_v19.py',
    'tests/test_prospective_validation_v21.py', 'tests/test_prospective_rules_v21.py')


def declared_blocks(spec):
    start = spec.get('seed_base')
    if not isinstance(start, int): return []
    if isinstance(spec.get('n_initials_per_scenario'), int):
        return [(start + i * 1000, spec['n_initials_per_scenario']) for i in range(len(spec['scenarios']))]
    count = spec.get('n_initials', spec.get('initials'))
    return [(start, count)] if isinstance(count, int) else []


def check_reserved_ranges():
    requested = [(63500001, CALIBRATION_N), (63600001, CONFIRMATION_N)]
    records = {}
    for path in sorted((ROOT / 'artifacts').rglob('protocol.json')):
        if 'source' in path.parts: continue
        data = json.loads(path.read_text(encoding='utf-8'))
        blocks = declared_blocks(data.get('spec', data))
        for start, count in blocks:
            if any(max(start, target) < min(start + count, target + size) for target, size in requested):
                raise ValueError('Reserved fresh initial range overlaps: ' + str(path))
        if blocks: records[path.relative_to(ROOT).as_posix()] = dict(blocks=blocks, sha256=digest(path))
    return records


def require_terminal(logs, protocol_sha256, shards):
    """Require the original supervised completion, before any outcome read."""
    logs = Path(logs)
    receipt_path = Path(str(logs) + '.supervisor_exit.json')
    if not receipt_path.exists():
        raise ValueError('Original supervisor has not exited; no outcomes may be read')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    if receipt.get('exit_code') != 0: raise ValueError('Original supervisor did not exit successfully')
    launch = json.loads((logs / 'launch.json').read_text())
    state = json.loads((logs / 'state.json').read_text())
    exits = json.loads((logs / 'worker_exits.json').read_text())['workers']
    if launch['protocol_sha256'] != protocol_sha256 or state['phase'] != 'completed':
        raise ValueError('Original study protocol or completion state differs')
    validate_worker_exits(exits, launch, shards)
    return dict(supervisor_exit_sha256=digest(receipt_path), worker_exits_sha256=digest(logs / 'worker_exits.json'),
                launch_sha256=digest(logs / 'launch.json'), completed_state_sha256=digest(logs / 'state.json'))


def freeze():
    if RULES.exists(): raise FileExistsError('Preserve the original prospective rules')
    _, phase = read_protocol(PHASE)
    require_committed([ROOT / name for name in SOURCES])
    prior = check_reserved_ranges()
    level = attempt_error(1)
    cutoffs = exact_cutoffs(CALIBRATION_N, RISK_LIMIT, level)
    power = []
    for adoption in (.1, .2, .3):
        for risk in (.01, .02, .03, .04, .05):
            p = single_target_power(CALIBRATION_N, adoption, risk, cutoffs)
            if risk == RISK_LIMIT and p > level + 1e-12: raise ValueError('Boundary-null error exceeded')
            power.append(dict(hypothetical_adoption=adoption, hypothetical_risk=risk, single_target_power=p,
                              all_six_pass_probability_lower_bound=joint_lower_bound([p] * 6)))
    plan = dict(purpose='Prospective statistical and complete-family selection rules, not a calibration result',
        candidate_families=list(FAMILIES), fixed_models=list(MODELS), models=phase['models'], modes=list(MODES),
        development_protocol_sha256=digest(PHASE / 'protocol.json'),
        development_selection='All three fixed models must have adoption >=20%, gain >=0.5%, a positive pointwise gain interval, no failed runs and empirical actual/bundle harm <=5%. Rank eligible complete families by worst six-target 95% development upper score, then minimum model gain, then phase05/phase10/repair order.',
        no_eligible_family='Record rejection and continue development; never splice models or treat baseline-only execution as success',
        calibration=dict(seed_base=63500001, independent_initials=CALIBRATION_N, sampling='iid_mixture',
            modes=list(MODES), mode_selection_stream=22107, physical_error_radius=.5,
            profiles='raw, nominal and one development-selected gate for each of all three fixed models, plus baseline',
            controller_count=10, actual_sumo_runs=CALIBRATION_N * 4 * 10),
        confirmation=dict(seed_base=63600001, independent_initials=CONFIRMATION_N, sampling='iid_mixture',
            modes=list(MODES), mode_selection_stream=22117, bootstrap_stream=22123, bootstrap_samples=5000,
            controller_count=10, actual_sumo_runs=CONFIRMATION_N * 4 * 10,
            actual_adoption_minimum=ADOPTION_MIN, mean_gain_percent_minimum=GAIN_PERCENT_MIN,
            all_three_gain_intervals_positive=True, all_three_primary_risks_certified=True),
        primary=dict(risk_limit=RISK_LIMIT, total_attempt_error=TOTAL_ATTEMPT_ERROR,
            allocation='delta_k = 0.05/[k(k+1)] for attempt k=1,2,...; every post-data attempt consumes its allocation',
            first_attempt_error=level, targets=['actual', 'bundle'], family_size=6,
            claim='All three fixed models and both conditional risks pass as one indivisible family',
            comparator_cannot_replace_failed_primary=True),
        comparators=dict(common_profiles=True, common_calibration_initials=CALIBRATION_N,
            conformal='Same frozen worst-mode/profile/car/metric residual construction; marginal coverage only',
            ltt='Same nine profiles; two-target Bonferroni exact-binomial calibration with comparison error 0.05',
            mean_spi='Same nine profiles; paired full-information Gaussian mean bands with comparison error 0.05, 50000 simulations, seed 22113',
            selections='Within each fixed training model; complete controller trajectories only',
            scope='Established task adaptations with distinct targets; not complete original SOTA reproductions and not evidence of outperforming LTT itself'),
        power_scenarios=power, power_assumptions='Six equal hypothetical marginals and a union lower bound; no target independence assumed; not a forecast of actual outcomes',
        historical_protocol_ranges=prior, current_outcomes_read=False, calibration_data_generated=False,
        calibration_execution_protocol_frozen=False, risk_certified=False,
        source_sha256={name:digest(ROOT / name) for name in SOURCES})
    RULES.mkdir(parents=True)
    save(RULES / 'plan.json', plan)
    for name in SOURCES:
        target = RULES / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    save(RULES / 'sha256.json', {p.relative_to(RULES).as_posix():digest(p) for p in RULES.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(rules_sha256=digest(RULES / 'plan.json'), calibration_initials=CALIBRATION_N,
        first_attempt_error=level, independent_confirmation_initials=CONFIRMATION_N,
        example_power=[r for r in power if r['hypothetical_adoption']==.2 and r['hypothetical_risk']==.02])), flush=True)


def read_rules():
    verify_manifest(RULES)
    plan = json.loads((RULES / 'plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT / name) != expected: raise ValueError('Frozen statistical/selection source changed: ' + name)
    return plan


def select(logs, attempt):
    plan = read_rules()
    evidence = require_terminal(logs, plan['development_protocol_sha256'], 3)
    _, phase = read_protocol(PHASE)
    verify_manifest(PHASE)
    summary = json.loads((PHASE / 'summary.json').read_text())
    if (summary['protocol_sha256'] != plan['development_protocol_sha256'] or
            summary['actual_sumo_runs'] != phase['spec']['expected_actual_sumo_runs'] or
            summary['independent_review_counts']['physical_traces'] != summary['actual_sumo_runs'] or
            phase['models'] != plan['models']):
        raise ValueError('Complete reviewed development identity differs')
    history = []
    for index in range(1, attempt):
        prior = ATTEMPTS / f'attempt_{index}'
        verify_manifest(prior)
        history.append(json.loads((prior / 'terminal.json').read_text()))
    allocation = validate_attempt_history(history, attempt)
    output = ATTEMPTS / f'attempt_{attempt}'
    if output.exists(): raise FileExistsError('Preserve previous attempt selection and data')
    selected = select_development_family(summary['results'], summary['independent_initials'])
    choice = selected['selected_family']
    output.mkdir(parents=True)
    save(output / 'selection.json', dict(**selected, **allocation, original_completion=evidence,
        common_profiles=list(profile_names(choice)) if choice else [],
        rules_sha256=digest(RULES / 'plan.json'), development_manifest_sha256=digest(PHASE / 'sha256.json'),
        development_summary_sha256=digest(PHASE / 'summary.json'),
        calibration_execution_protocol_still_required=True, new_calibration_data_generated=False))
    print(json.dumps(dict(selected_family=choice, attempt=attempt, risk_certified=False,
        calibration_execution_protocol_still_required=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    selector = sub.add_parser('select')
    selector.add_argument('logs')
    selector.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'freeze': freeze()
    else: select(args.logs, args.attempt)
