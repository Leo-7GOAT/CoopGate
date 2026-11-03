"""Export terminal-complete scientific inputs for the separate document runtime.

The document builder need not import PyTorch or the simulation environment.
No report is fabricated, rounded, re-tested, selected or read from a live study.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from mappo_completed_powered_attempt5 import complete_evidence
import mappo_powered_history_formal_experiment as e

INPUTS = {}


def read(path):
    path = Path(path)
    value = json.loads(path.read_bytes())
    INPUTS[path.relative_to(ROOT).as_posix()] = e.sha(path)
    return value


def owned_report(directory, count_key=None, count=None):
    """Validate actual native terminal evidence before accessing report effects."""
    directory = Path(directory)
    owner = read(directory / 'owner_native_exit.json')
    if owner['actual_native_exit_code'] != 0:
        raise ValueError('Native owner did not exit zero: ' + str(directory))
    if owner['report_sha256'] != e.sha(directory / 'report.json'):
        raise ValueError('Report is not bound to this actual native completion')
    report = read(directory / 'report.json')
    if report.get('completed', report.get('complete')) is not True:
        raise ValueError('Incomplete report: ' + str(directory))
    if count_key and report[count_key] != count:
        raise ValueError('Original fixed execution scope changed')
    children = report.get('children', [])
    for child in (children.values() if isinstance(children, dict) else children):
        if child['actual_native_exit_code'] != 0:
            raise ValueError('A child native execution failed')
    return report


def build(out):
    if out.exists():
        raise FileExistsError('Retain all evidence bundle versions')
    cert, comparison = complete_evidence()
    # Scheduler root itself must have an observed successful native terminal.
    pipeline = owned_report(e.BASE / 'powered_history_completion_pipeline_20261103',
                            'supplementary_actual_sumo_executions', 6144)
    if (pipeline['primary_actual_sumo_executions'] != 45056
            or pipeline['independent_initials'] != 5120
            or pipeline['primary_certificate_sha256'] != e.sha(e.FORMAL / 'analysis/certificate.json')):
        raise ValueError('Completed pipeline describes another primary study')
    data = dict(certificate=read(e.FORMAL / 'analysis/certificate.json'),
                comparison=read(e.FORMAL / 'analysis/comparisons.json'),
                primary_completion=read(e.FORMAL / 'completion_receipt.json'),
                primary_native_owner=read(e.FORMAL / 'orchestrator_native_receipt.json'),
                primary_plan=read(e.FORMAL / 'plan.json'), pipeline=pipeline)
    if data['certificate'] != cert or data['comparison'] != comparison:
        raise ValueError('Primary analysis changed during export')
    data['mechanism_ablations'] = {}
    for variant in ('compact', 'phase'):
        directory = e.BASE / ('powered_history_' + variant + '_holdout_ablation_20261103')
        report = owned_report(directory, 'actual_additional_sumo_executions', 3072)
        if (report['comparator_variant'] != variant or report['original_confirmation_initials'] != 1024
                or report['new_independent_initials'] != 0 or report['new_error_allocated']
                or report['primary_certificate_sha256'] != e.sha(e.FORMAL / 'analysis/certificate.json')):
            raise ValueError('Wrong fixed paired ablation identity')
        read(directory / 'plan.json')
        if report['paired_outcomes_sha256'] != e.sha(directory / 'paired_outcomes.json'):
            raise ValueError('Paired mechanism observations changed')
        INPUTS[(directory / 'paired_outcomes.json').relative_to(ROOT).as_posix()] = e.sha(directory / 'paired_outcomes.json')
        data['mechanism_ablations'][variant] = report
    for label, relative, key, count in (
        ('random_proposals', 'development/random_proposal_ablation_20261103', 'actual_additional_sumo_executions', 960),
        ('history_development', 'development/observed_history_validation_20261103', 'actual_sumo_executions', 644),
        ('phase_development', 'development/strict_phase_ablation_20261103', 'actual_sumo_executions', 276),
        ('power_pilot', 'development/history_power_pilot_20261103', 'actual_sumo_executions', 1280),
        ('power_design', 'history_power_design_20261103', 'pilot_initials', 320),
        ('deployed_roles', 'deployed_role_participation_20261103', 'new_sumo_runs', 0),
        ('identification_limits', 'lag_identification_limits_20261103', 'new_sumo_runs', 0)):
        data[label] = owned_report(e.BASE / relative, key, count)
    random = data['random_proposals']
    if (random['new_independent_initials'] != 0 or random['actor_checkpoints_used']
            or random['statistical_certificate'] or random['new_error_allocated']
            or set(random['results']) != {'random_0', 'random_1', 'random_2'}):
        raise ValueError('Random proposal control must remain policy-free development')
    old = e.BASE / 'formal_mappo_attempt4_20261102'
    old_owner = read(old / 'orchestrator_native_receipt.json')
    if (old_owner['actual_native_exit_code'] != 0
            or old_owner['completion_receipt_sha256'] != e.sha(old / 'completion_receipt.json')):
        raise ValueError('Complete original failed study must remain evidenced')
    old_completion = read(old / 'completion_receipt.json')
    if old_completion['total_actual_sumo_runs'] != 37888:
        raise ValueError('Original failed study scope changed')
    for name in ('certificate', 'comparisons'):
        if old_completion[name + '_sha256'] != e.sha(old / ('analysis/' + name + '.json')):
            raise ValueError('Original study analysis changed')
    data['prior_attempt4'] = dict(certificate=read(old / 'analysis/certificate.json'),
                                  comparison=read(old / 'analysis/comparisons.json'))
    recovery = owned_report(e.BASE / 'coupled_holdout_ablation_recovery_20261103',
                            'complete_recorded_old_gate_outcomes', 3072)
    if recovery['actual_resumed_sumo_executions'] != 1697 or not recovery['original_root_exit_unknown']:
        raise ValueError('Original interrupted recovery accounting must remain explicit')
    data['prior_paired_recovery'] = recovery
    data['diagnosis'] = read(e.BASE / 'formal_mappo_attempt5_powered_diagnosis_20261103/report.json')
    if (data['diagnosis']['original_records'] != 45056
            or data['diagnosis']['independent_initials'] != 5120
            or not data['diagnosis']['complete_statistics_independently_match']):
        raise ValueError('Whole-study diagnostic scope incomplete')
    data['figures'] = read(e.BASE / 'formal_mappo_attempt5_powered_figures_20261103/report.json')
    if (data['figures']['source_certificate_sha256'] != e.sha(e.FORMAL / 'analysis/certificate.json')
            or data['figures']['source_comparisons_sha256'] != e.sha(e.FORMAL / 'analysis/comparisons.json')):
        raise ValueError('Figures describe another exact scientific decision')
    for image in data['figures']['figures']:
        path = e.BASE / 'formal_mappo_attempt5_powered_figures_20261103' / image['path']
        if e.sha(path) != image['sha256']:
            raise ValueError('A completed figure was modified')
        INPUTS[path.relative_to(ROOT).as_posix()] = image['sha256']
    data['training'] = {}
    for seed in (4703, 4704, 4705):
        audit = read(e.BASE / f'development/joint_{seed}/audit.json')
        if (audit['algorithm'] != 'mappo_ctde_20261101' or audit['steps'] != 2097152
                or not audit['final_budget_complete'] or not audit['on_policy_proposals_verified']
                or not audit['independent_gae_verified']):
            raise ValueError('A genuine complete joint MAPPO training audit is missing')
        data['training'][str(seed)] = audit
    data['single_competence'] = read(e.BASE / 'development/single_4601_sumo_full/single_competence.json')
    data['retained_single_failure'] = read(e.BASE / 'development/single_4603_exact/single_competence.json')
    if not data['single_competence']['numeric_thresholds_met']:
        raise ValueError('Verified single-vehicle parent competence is missing')
    INPUTS.update(data['primary_plan']['source_sha256'])
    data['scope_notes'] = dict(
        primary_independent_initials=5120, paired_ablation_new_initials=0,
        random_control_new_initials=0, primary_actual_sumo_executions=45056,
        fixed_paired_ablation_actual_sumo_executions=6144,
        separate_development_random_actual_sumo_executions=960,
        no_new_tests_or_error_allocated=True, no_selection_on_primary_effects=True,
        raw_mappo_superiority_over_strong_baseline_proven=False,
        causal_central_critic_advantage_proven=False, full_box_performance_theorem=False,
        hard_real_time_or_real_vehicle_validation=False,
        gain_t_bound_normal_iid_exact_otherwise_asymptotic=True,
        mechanism_comparisons_descriptive_not_new_certificates=True)
    data['inputs_sha256'] = dict(sorted(INPUTS.items()))
    data['exporter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data['created_utc'] = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(out), source_inputs=len(INPUTS),
                         joint_deployment_certified=cert['joint_deployment_certified'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(dict(required_terminal_witness_files_present=all(p.is_file() for p in (
            e.FORMAL / 'orchestrator_native_receipt.json',
            e.BASE / 'powered_history_completion_pipeline_20261103/owner_native_exit.json',
            e.BASE / 'development/random_proposal_ablation_20261103/owner_native_exit.json')),
            scientific_outcomes_read=False)), flush=True)
    elif args.out:
        build(args.out)
    else:
        parser.error('--out or --check is required')
