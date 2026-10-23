"""Unchanged recovery gates on the complete original scene/actuator grid."""
import argparse
import json
from pathlib import Path

import study_recovery_gate_v35 as core
from execute_prospective_v23 import exclusive_json, runtime_identity, sources_committed, utc
from profile_phase_runtime_v24 import committed, digest

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = core.OUT
OUT = ROOT/'artifacts/v36/recovery_gate_complete_grid'
SCRIPT = 'tools/extend_recovery_gate_v36.py'
TEST = 'tests/test_recovery_gate_extension_v36.py'


def full_groups(cases, failures):
    if (any(len(k) != 6 or k[-1] != 'actuator_recovery' for k in cases)
            or any(len(k) != 6 or k[-1] != 'baseline' for k in failures)):
        raise ValueError('Original recovery conditions and failure identities required')
    core.recovery.complete_cases([list(k[:5])+['baseline'] for k in cases])
    conditions = {tuple(k[:5]) for k in cases}
    failed = {tuple(k[:5]) for k in failures}
    if len(failures) != 10 or len(failed) != 10 or not failed <= conditions:
        raise ValueError('All ten old failures must remain in the full grid')
    methods = [('baseline','none')]+[(r,m) for m in core.MODELS for r in core.RULES[1:]]
    result = []
    for index,condition in enumerate(sorted(conditions,key=lambda k:(k[2],k[1],k[4],k[0]))):
        offset = index % len(methods)
        order = methods[offset:]+methods[:offset]
        if index % 2:
            order = order[::-1]
        result.append(dict(block_index=index,condition=list(condition),old_baseline_failed=condition in failed,
            cases=[list(condition)+list(method) for method in order]))
    return result


def freeze():
    prior = core.inputs()
    report = core.read(PREFLIGHT/'analysis/report.json')
    proof = core.read(PREFLIGHT/'completion_verification.json')
    if (proof['summary_exit_code'] != 0 or proof['worker_exit_code'] != 0
            or proof['report_sha256'] != digest(PREFLIGHT/'analysis/report.json')
            or not report['all_physical_and_full_gate_replays_match'] or report['actual_sumo_runs'] != 390):
        raise ValueError('Complete successful integration and every gate replay are required')
    original = core.read(core.recovery.OUT/'plan.json')
    groups = full_groups(original['cases'],original['baseline_failed_cases'])
    sources = dict(prior['source_sha256'])
    sources.update({p:digest(ROOT/p) for p in (SCRIPT,TEST)})
    for name,expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen recovery implementation changed: '+name)
    sources_committed([ROOT/p for p in sources])
    paths = [PREFLIGHT/n for n in ('plan.json','analysis/report.json','completion_verification.json')]
    files = dict(prior['inputs_sha256'])
    files.update({p.relative_to(ROOT).as_posix():digest(p) for p in paths})
    OUT.mkdir(parents=True,exist_ok=False)
    exclusive_json(OUT/'plan.json',dict(utc=utc(),runtime=runtime_identity(),source_sha256=sources,
        inputs_sha256=files,models=prior['models'],methods=prior['methods'],groups=groups,
        blocks=640,actual_sumo_runs=8320,old_unique_initials=160,new_independent_initials=0,
        new_risk_certificate=False,workers=1,
        scope='All five original scenes, all four actuator classes and all 32 initials per scene; all ten original baseline failures. The 390 preflight conditions/methods are rerun unchanged and remain repeats, not independent samples.',
        implementation='Exactly the frozen V35 gates and auditors; same three trained actor sets, known class bounds, online intervals, physical oracle, candidate construction, ranking, one/33 paths and thresholds. No new training, tuning or policy selection.',
        order='Original within-scene index, scene, plant, then fixed rotating/reversing method order. All 13 method/model combinations in every condition.',
        physical_comparison=prior['method'],prior_adapters=prior['prior_adapters'],
        online_information=prior['online_information'],prediction=prior['prediction'],
        performance_rule=prior['performance_rule'],adoption=prior['adoption'],harm=prior['harm'],
        review='Store every actual execution before checking. Reconstruct every physical substep and baseline command, match every common baseline trajectory to V34, then replay every full gate decision and terminal exception after complete successful actual exit.',
        outcomes=prior['outcomes'],timing=prior['timing'],
        statistical_scope='All 160 old development initials, shared across four plants and three models. This tests the complete existing scenario scope; no new independent efficacy confirmation, risk certificate, equal-compute claim or hard-real-time assertion.',
        ongoing_calibration_changed=False,ongoing_v32_changed=False,partial_formal_outcomes_read=False))
    print(json.dumps(dict(blocks=640,actual_sumo_runs=8320,plan_sha256=digest(OUT/'plan.json'))),flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = core.read(OUT/'plan.json')
    for name,expected in {**plan['source_sha256'],**plan['inputs_sha256']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen full-grid source/input changed: '+name)
    sources_committed([ROOT/p for p in plan['source_sha256']])
    if runtime_identity() != plan['runtime']:
        raise ValueError('Frozen runtime changed')
    original = core.read(core.recovery.OUT/'plan.json')
    if full_groups(original['cases'],original['baseline_failed_cases']) != plan['groups']:
        raise ValueError('The entire original grid must remain fixed')
    for item in plan['models'].values():
        if digest(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen trained model changed')
    return plan


def metadata(group):
    return dict(plan_sha256=digest(OUT/'plan.json'),block_index=group['block_index'],stage='full_recovery_gate_grid')


def dispatch(command):
    if command == 'freeze':
        return freeze()
    # Reuse the unchanged collection/replay code with an explicit output and
    # complete-scope validator. This changes no source file or active study.
    core.OUT,core.SCRIPT,core.inputs,core.metadata = OUT,SCRIPT,inputs,metadata
    return getattr(core,command)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','worker','watch','summarize'))
    dispatch(parser.parse_args().command)
