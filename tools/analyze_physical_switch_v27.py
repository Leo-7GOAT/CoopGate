"""Complete-data analysis of common-condition physical and performance gates.

Freeze this analysis before reading the ongoing full comparison. Both actual
successful exits and all complete blocks are required; calibration is untouched.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from coopgate.common_adoption_v27 import changed_active_commands
from coopgate.episode_store_v21 import observations
from coopgate.physical_comparison_analysis_v27 import (
    analyze_phase, analyze_transfer, annotated, expected_cases, identity)
from execute_prospective_v23 import exclusive_json, runtime_identity, sources_committed, utc
from profile_phase_runtime_v24 import committed, digest
import audit_common_adoption_v27 as common
import audit_physical_switch_v27 as physical

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v27/physical_switch_analysis'
SOURCES = ('src/coopgate/physical_comparison_analysis_v27.py',
           'tools/analyze_physical_switch_v27.py',
           'tests/test_physical_comparison_analysis_v27.py',
           'tests/test_physical_comparison_loader_v27.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def method_lists(plan):
    additions = [f'{rule}_{model}' for model in plan['models'] for rule in plan['rules']]
    return {study: list(protocol['spec']['methods'])+additions
            for study, protocol in plan['protocols'].items()}


def freeze():
    comparison = read(physical.OUT/'plan.json')
    adoption = read(common.OUT/'plan.json')
    for path in (physical.OUT/'plan.json', common.OUT/'plan.json'):
        committed(path)
    sources_committed([ROOT/name for name in SOURCES])
    if adoption['comparison_plan_sha256'] != digest(physical.OUT/'plan.json'):
        raise ValueError('Adoption audit belongs to a different comparison')
    methods = method_lists(comparison)
    grid = {s: len(expected_cases(s, p['spec'], methods[s])) for s, p in comparison['protocols'].items()}
    if grid != dict(phase=9728, transfer=16000) or adoption['old_episodes'] != 18816:
        raise ValueError('Full old/new comparison grid required')
    plan = dict(purpose=__doc__, utc=utc(), source_sha256={name:digest(ROOT/name) for name in SOURCES},
        comparison_plan_sha256=digest(physical.OUT/'plan.json'),
        adoption_plan_sha256=digest(common.OUT/'plan.json'), runtime=runtime_identity(),
        methods=methods, expected_episodes=grid, old_episodes=18816, new_comparison_executions=6912,
        expected_phase_methods=19, expected_transfer_cells=500,
        expected_primary_contrasts=dict(phase=6, transfer=120),
        expected_independent_initials=dict(phase=128, transfer=160),
        bootstrap_samples=comparison['bootstrap_samples'], bootstrap_stream=comparison['bootstrap_stream'],
        bootstrap='For each cell or phase subset with n initials, default_rng(27113).integers(n, size=(5000,n)); same draws for every matched method in that cell. Pointwise percentile 95% mean-cost intervals only.',
        phase_actual='Exactly one mode per initial from the original seed_base and mode_selection_stream; identical mode for every method.',
        phase_bundle='At initial level: A=any mode adopts, H=any mode harms; count A & H even when adoption and harm occur in different modes.',
        primary_contrasts='For each fixed model, phase05 versus mamps_shared and mps_shared. Positive cost increment=comparator cost minus phase05 cost. Risk difference=phase05 conditional risk minus comparator conditional risk.',
        original_events='Original learned_action_steps retained alongside common actual-command events; no old certification event or result is overwritten.',
        pairing='Match seed, initial state, mode, radius and, for transfer, plant parameters and external innovation tape. Across plants match initial and external innovation; do not require realized actuator errors to match.',
        exclusions='None. Retain capped unsuccessful outcomes, terminal errors, all fixed models, baseline failures and error-bound exceedances.',
        conditional_risk='Descriptive P_hat(H|A); zero adoption is null, never zero. No risk difference if either denominator is zero.',
        timing='Only new complete callbacks, including terminal error calls. Historical callback times are from different execution periods and are not a speed comparison.',
        scope='Previously used development/transfer initials only. Modes, methods, models and plants do not add independent initials. No overall winner selection, simultaneous-coverage claim, new risk certificate or calibration outcome read.',
        new_sumo_runs_from_analysis=0, new_independent_initials=0, new_risk_certificate=False,
        active_calibration_outcomes_read=False)
    OUT.mkdir(parents=True, exist_ok=False)
    exclusive_json(OUT/'plan.json', plan)
    print(json.dumps(dict(plan_sha256=digest(OUT/'plan.json'), expected_episodes=grid)), flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    if (plan['comparison_plan_sha256'] != digest(physical.OUT/'plan.json')
            or plan['adoption_plan_sha256'] != digest(common.OUT/'plan.json')
            or plan['runtime'] != runtime_identity()):
        raise ValueError('Analysis dependencies or runtime changed')
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen analysis source changed: '+name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    return plan


def completion_gate(plan, physical_root=None, common_root=None, commit_check=committed):
    """Do not open observations until complete native exit evidence is present."""
    physical_root = physical.OUT if physical_root is None else Path(physical_root)
    common_root = common.OUT if common_root is None else Path(common_root)
    paths = dict(physical_exit=physical_root/'full/execution/exit.json',
        physical_launch=physical_root/'full/execution/launch.json',
        physical_done=physical_root/'full/execution/collection_complete.json',
        physical_verification=physical_root/'full/analysis/verification.json',
        common_exit=common_root/'execution/observation.json',
        common_start=common_root/'execution/start.json',
        common_done=common_root/'execution/collection_complete.json',
        common_report=common_root/'analysis/report.json')
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise ValueError('Complete successful executions and reviews required; missing: '+'; '.join(missing))
    records = {name:read(path) for name, path in paths.items()}
    sha = plan['comparison_plan_sha256']
    physical.validate_completion(dict(actual_sumo_runs=6912, blocks=288), sha,
        records['physical_exit'], records['physical_launch'], records['physical_done'])
    proof = records['physical_verification']
    if (proof['stage'] != 'full' or proof['actual_sumo_runs'] != 6912 or proof['plan_sha256'] != sha
            or proof['exit_receipt_sha256'] != digest(paths['physical_exit'])):
        raise ValueError('Complete physical verification differs')
    sha = plan['adoption_plan_sha256']
    start, done, exit_ = (records[k] for k in ('common_start', 'common_done', 'common_exit'))
    report = records['common_report']
    if (exit_.get('exit_code') != 0 or exit_.get('native_pid') != start['native_pid']
            or any(r['plan_sha256'] != sha for r in (start, done, exit_, report))
            or done['native_pid'] != start['native_pid'] or done['old_episodes'] != 18816 or done['blocks'] != 288
            or report['old_episodes'] != 18816 or report['exit_observation_sha256'] != digest(paths['common_exit'])):
        raise ValueError('Complete historical common-event audit differs')
    commit_check(paths['physical_verification'])
    commit_check(paths['common_report'])
    return {name:digest(path) for name, path in paths.items()}


def validate_common_record(group, case, record, entry):
    row = record['original_row']
    if (list(case) != entry['case'] or identity(group['study'], row) != case
            or common.object_hash(row) != entry['source_row_sha256']
            or record['source_row_sha256'] != entry['source_row_sha256']
            or record['source_trace_path'] != entry['trace_path']
            or record['source_trace_sha256'] != entry['trace_sha256']):
        raise ValueError('Historical audit must preserve frozen source row and case')
    if record['original_learning_steps'] != row['learned_action_steps']:
        raise ValueError('Original event was overwritten')
    result = annotated(row, record['common_learning_steps'])
    if record['common_episode_adopted'] != (result['common_learning_steps'] > 0):
        raise ValueError('Common event count differs')
    return result


def validate_new_record(case_with_rule, record, trace, study):
    row = record['row']
    name = case_with_rule[-1]+'_'+case_with_rule[-2].rsplit('_', 1)[1]
    expected = (*case_with_rule[:-2], name)
    if identity(study, row) != expected or row['physical_switch_rule'] != case_with_rule[-1]:
        raise ValueError('Physical comparison case or method differs')
    physical.check_calls(row, trace, record['calls'])
    if len(trace) != row['control_steps']:
        raise ValueError('Missing executed decisions')
    state = np.asarray(row['initial'])
    count = 0
    for item in trace:
        active = bool(changed_active_commands(state, item['command'], item['common_baseline']).any())
        if active != item['actual_learning_used']:
            raise ValueError('Physical mode label differs from actual-command adoption')
        count += int(active)
        state = np.asarray(item['x'])
    if row['learned_action_steps'] != count:
        raise ValueError('Physical episode adoption differs')
    return annotated(row, count)


def load_rows(plan):
    # Recheck both input plans, frozen sources, complete seals and per-record
    # contents rather than trusting counts in a presentation file.
    old_plan, new_plan = common.inputs(), physical.inputs('full')
    rows = dict(phase=[], transfer=[])
    entries = {(g['block_index'], tuple(e['case'])):e for g in old_plan['groups'] for e in g['cases']}
    for group, case, record, trace in common.complete_records(old_plan):
        rows[group['study']].append(validate_common_record(group, case, record, entries[group['block_index'], case]))
    if sum(map(len, rows.values())) != plan['old_episodes']:
        raise ValueError('All historical episodes required')
    timings = defaultdict(lambda:dict(wall_ms=[], error_calls=0, executions=0))
    new_runs = commands = terminal_failures = 0
    for group in physical.groups_for(new_plan, 'full'):
        directory = physical.OUT/'full/blocks'/f"{group['block_index']:05d}"
        done = read(directory/'complete.json')
        schedule = physical.schedule(group)
        if (done['database_sha256'] != digest(directory/'observations.sqlite')
                or done['seal_sha256'] != digest(directory/'observations.sqlite.seal.json')
                or done['actual_sumo_runs'] != len(schedule) or len(done['reviews']) != len(schedule)):
            raise ValueError('Physical block or independent review changed')
        data = observations(directory/'observations.sqlite', physical.metadata('full', group), schedule)
        for (case, record, trace), review in zip(data, done['reviews'], strict=True):
            row = validate_new_record(case, record, trace, group['study'])
            if (review['physical_traces'] != 1 or review['reconstructed_commands'] != len(trace)
                    or review['terminal_failures'] != int(row['aborted'])
                    or not review['initial_and_external_noise_matched']
                    or review['actual_learning_steps'] != row['common_learning_steps']):
                raise ValueError('Independent physical review scope differs')
            rows[group['study']].append(row)
            item = timings[group['study'], row['controller']]
            item['wall_ms'].extend(c['wall_ms'] for c in record['calls'])
            item['error_calls'] += sum(c['error'] is not None for c in record['calls'])
            item['executions'] += 1
            new_runs += 1
            commands += len(trace)
            terminal_failures += int(row['aborted'])
    if new_runs != plan['new_comparison_executions'] or {s:len(r) for s, r in rows.items()} != plan['expected_episodes']:
        raise ValueError('Full merged comparison grid required')
    timing_report = []
    for (study, method), item in sorted(timings.items()):
        values = np.asarray(item['wall_ms'])
        timing_report.append(dict(study=study, method=method, executions=item['executions'],
            callbacks=len(values), terminal_error_calls=item['error_calls'],
            mean_ms=float(values.mean()), p95_ms=float(np.percentile(values, 95)),
            p99_ms=float(np.percentile(values, 99)), maximum_ms=float(values.max()),
            callbacks_over_500ms=int((values > 500).sum())))
    return rows, new_plan, dict(new_execution_commands=commands, terminal_failures=terminal_failures,
                                new_callbacks=timing_report)


def number(value, places=4):
    return 'undefined' if value is None else f'{value:.{places}f}'


def interval(values):
    return '['+', '.join(number(v, 6) for v in values)+']'


def display_report(report):
    phase, transfer = report['phase'], report['transfer']
    lines = ['# Common-condition physical and performance gate comparison', '',
        'All 18,816 historical executions and 6,912 added physical-gate executions are retained. '
        'They reuse 128 development and 160 transfer initials. No new independent efficacy initials or risk certificate.', '',
        'Adoption means at least one active vehicle executed a command different from the same-state common baseline. '
        'Harm means any vehicle cost/arrival exceeds the full-baseline outcome, or execution failure. '
        'Zero-adoption conditional risk is undefined; original event counts are retained separately.', '',
        'All intervals below are pointwise, descriptive paired-initial mean-cost bootstrap intervals. '
        'They do not provide simultaneous coverage or conditional-risk bounds. '
        'MAMPS/MPS are common-oracle/common-baseline adaptations, not original benchmark reproductions.', '',
        '## Development: one preassigned actual mode per initial', '',
        '| Method | Gain % | Gain 95% interval (s) | Adopted / N | Harmful / adopted | Original adopted | Failures | Bundle harmful / adopted |',
        '|---|---:|---|---|---|---:|---:|---|']
    for item in phase['methods']:
        r, b = item['actual'], item['four_mode_bundle']
        lines.append(f"| {item['method']} | {number(r['mean_gain_percent'])} | {interval(r['gain_pointwise_bootstrap95'])} | "
            f"{r['adopted']}/{r['independent_initials']} | {r['harmful_adopted']}/{r['adopted']} | "
            f"{r['original_adopted']} | {r['failed']} | {b['harmful_adopted']}/{b['adopted']} |")
    lines += ['', 'The JSON also contains every ordinary/dense subgroup, every vehicle outcome, '
              'all-mode failure counts and the exact preassigned mode of every initial.', '',
              '## Primary paired contrasts', '',
              'Positive cost increment favors phase05. Negative risk difference favors phase05. '
              'A lower risk alone is not a dominance claim if adoption or gain decreases.', '',
              '| Study | Scene | Plant | Model | Comparator | Cost increment (s) | Pointwise 95% interval | Adoption difference | Conditional risk difference |',
              '|---|---|---|---:|---|---:|---|---:|---:|']
    for study, contrasts in (('phase', phase['primary_contrasts']), ('transfer', transfer['primary_contrasts'])):
        for r in contrasts:
            lines.append(f"| {study} | {r.get('scenario', 'mixture')} | {r.get('plant', 'additive')} | {r['model']} | "
                f"{r['comparator']} | {number(r['mean_increment_seconds'], 6)} | {interval(r['increment_pointwise_bootstrap95'])} | "
                f"{number(r['adoption_rate_difference'])} | {number(r['empirical_conditional_risk_difference'])} |")
    lines += ['', '## All 500 transfer cells', '',
        'Each cell has 32 matched initials. Models and plants do not multiply the independent sample count. '
        'Baseline failures and physical error exceedances remain visible. No safety claim covers motion after an abort.', '',
        '| Scene | Plant | Method | Gain % | Gain 95% interval (s) | Adopted / N | Harmful / adopted | Original adopted | Failed | Baseline failed | Error-exceeded |',
        '|---|---|---|---:|---|---|---|---:|---:|---:|---:|']
    for r in transfer['cells']:
        lines.append(f"| {r['scenario']} | {r['plant']} | {r['method']} | {number(r['mean_gain_percent'])} | "
            f"{interval(r['gain_pointwise_bootstrap95'])} | {r['adopted']}/{r['independent_initials']} | "
            f"{r['harmful_adopted']}/{r['adopted']} | {r['original_adopted']} | {r['failed']} | "
            f"{r['baseline_failed']} | {r['exceeded_error_runs']} |")
    lines += ['', '## Added physical-gate callback timing', '',
        'Includes terminal error calls. Concurrent desktop load; old controller timings are not paired speed comparisons. '
        'These callback observations do not establish hard-real-time or physical actuation-delay safety.', '',
        '| Study | Method | Executions | Calls | Error calls | P95 ms | Maximum ms | Over 500 ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in report['execution_checks']['new_callbacks']:
        lines.append(f"| {r['study']} | {r['method']} | {r['executions']} | {r['callbacks']} | {r['terminal_error_calls']} | "
                     f"{number(r['p95_ms'], 3)} | {number(r['maximum_ms'], 3)} | {r['callbacks_over_500ms']} |")
    return '\n'.join(lines)+'\n'


def analyze():
    plan = inputs()
    evidence = completion_gate(plan)
    rows, comparison, checks = load_rows(plan)
    args = dict(models=list(comparison['models']), rules=comparison['rules'],
                samples=plan['bootstrap_samples'], stream=plan['bootstrap_stream'])
    phase = analyze_phase(rows['phase'], comparison['protocols']['phase']['spec'], plan['methods']['phase'], **args)
    transfer = analyze_transfer(rows['transfer'], comparison['protocols']['transfer']['spec'], plan['methods']['transfer'], **args)
    if (len(phase['methods']) != 19 or len(phase['primary_contrasts']) != 6
            or len(transfer['cells']) != 500 or len(transfer['primary_contrasts']) != 120):
        raise ValueError('Complete report scope required')
    report = dict(plan_sha256=digest(OUT/'plan.json'), utc=utc(), input_evidence_sha256=evidence,
        old_episodes=18816, added_comparison_executions=6912, phase=phase, transfer=transfer,
        execution_checks=checks, new_sumo_runs_from_analysis=0, new_independent_initials=0,
        new_risk_certificate=False, original_certificates_changed=False, active_calibration_outcomes_read=False)
    output = OUT/'results'
    output.mkdir(exist_ok=False)
    exclusive_json(output/'report.json', report)
    (output/'report.md').write_text(display_report(report), encoding='utf-8')
    exclusive_json(output/'sha256.json', {p.name:digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(dict(phase_methods=19, transfer_cells=500, primary_contrasts=126,
                         new_risk_certificate=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'check-ready', 'run'))
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'check-ready':
        print(json.dumps(completion_gate(inputs())), flush=True)
    else:
        analyze()
