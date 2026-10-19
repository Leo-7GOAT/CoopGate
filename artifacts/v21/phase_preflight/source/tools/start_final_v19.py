"""Enforce completed, committed calibration before any final-test execution."""
import argparse
import json
import subprocess
from pathlib import Path

from compare_established_v19 import complete_study, read_plan
from probe_disturbed_gate_v19 import digest, save
from study_gate_v19 import read_protocol, worker, finalize

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'artifacts/v19/comparison_plan'
CAL = ROOT / 'artifacts/v19/comparison_calibration'
RISK = ROOT / 'artifacts/v19/risk_validation'
FINAL = ROOT / 'artifacts/v19/final_confirmation'
LAUNCH = ROOT / 'artifacts/v19/final_launch.json'


def validate_identity(plan, calibration, risk, final):
    if risk['spec'].get('data_role') != 'risk_validation' or final['spec'].get('data_role') != 'final_confirmation':
        raise ValueError('Calibration/final data roles must remain distinct')
    if risk['models'] != final['models'] or calibration['models'] != risk['models']:
        raise ValueError('Training model identity changed between splits')
    if risk['spec']['methods'] != final['spec']['methods'] or plan['profile_definitions'] != final['spec']['methods']:
        raise ValueError('Complete controller profile definitions changed')
    for path, expected in risk['source_sha256'].items():
        if final['source_sha256'].get(path) != expected:
            raise ValueError('Runtime policy or executor source changed: ' + path)
    for key in ('seed_base', 'n_initials', 'sampling', 'modes'):
        if final['spec'][key] != plan['final_test'][key]:
            raise ValueError('Frozen final-test design changed: ' + key)
    if risk['spec']['sampling'] != final['spec']['sampling'] or risk['spec']['modes'] != final['spec']['modes']:
        raise ValueError('Calibration and test populations differ')
    risk_ids = set(range(risk['spec']['seed_base'], risk['spec']['seed_base'] + risk['spec']['n_initials']))
    final_ids = set(range(final['spec']['seed_base'], final['spec']['seed_base'] + final['spec']['n_initials']))
    if risk_ids & final_ids:
        raise ValueError('Calibration/test initial IDs overlap')
    names = plan['profiles']
    if calibration['names'] != names or set(names) != set(final['spec']['methods']) - {'baseline'}:
        raise ValueError('Frozen candidate family changed')
    for rule in ('ltt_selection', 'mean_spi_selection'):
        if set(calibration[rule]) != set(plan['seeds']):
            raise ValueError('A training replicate is missing')
        for seed, name in calibration[rule].items():
            if name != 'baseline' and (name not in names or not name.endswith('_' + seed)):
                raise ValueError('Selection crosses a training replicate or introduces a new profile')
    return dict(calibration_initials=len(risk_ids), final_initials=len(final_ids),
                disjoint_initials=True, complete_policy_identity_preserved=True)


def verify_manifest(directory):
    directory = Path(directory)
    for path, expected in json.loads((directory / 'sha256.json').read_text()).items():
        if digest(directory / path) != expected:
            raise ValueError('Archived artifact changed: ' + str(directory / path))


def git_output(*args):
    result = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def require_committed(paths):
    relative = [Path(p).resolve().relative_to(ROOT).as_posix() for p in paths]
    for path in relative:
        git_output('ls-files', '--error-unmatch', '--', path)
    if git_output('status', '--porcelain', '--', *relative):
        raise ValueError('Calibration/launch records must be committed and unchanged before final execution')


def inputs():
    plan = read_plan(PLAN)
    verify_manifest(CAL)
    calibration = json.loads((CAL / 'calibration.json').read_text())
    _, risk = read_protocol(RISK)
    _, final = read_protocol(FINAL)
    identities = validate_identity(plan, calibration, risk, final)
    if calibration['plan_sha256'] != digest(PLAN / 'plan.json'):
        raise ValueError('Calibration was produced by a different comparison plan')
    if calibration['calibration_protocol_sha256'] != digest(RISK / 'protocol.json'):
        raise ValueError('Calibration source protocol changed')
    if calibration['calibration_episodes_sha256'] != digest(RISK / 'episodes.json'):
        raise ValueError('Calibration source outcomes changed')
    if calibration['conditional_risk_sha256'] != digest(CAL / 'conditional_risk.json'):
        raise ValueError('Conditional risk result changed')
    if final['spec']['comparison_plan_sha256'] != digest(PLAN / 'plan.json'):
        raise ValueError('Final protocol binds a different comparison plan')
    files = [CAL / 'calibration.json', CAL / 'conditional_risk.json', CAL / 'sha256.json',
             PLAN / 'plan.json', FINAL / 'protocol.json']
    require_committed(files)
    return calibration, identities, files


def prepare():
    if LAUNCH.exists() or any(FINAL.glob('shard_*')):
        raise FileExistsError('Do not create a new launch record after preparation or final-test execution')
    calibration, identities, files = inputs()
    # Full finalization and integrity, never a count from a live shard file.
    complete_study(RISK, 'risk_validation')
    record = dict(purpose='Immutable prerequisite record for untouched final execution',
                  calibration_commit=git_output('rev-parse', 'HEAD'),
                  identities=identities, both_targets_certified=calibration['both_targets_certified'],
                  input_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in files},
                  launcher_sha256=digest(__file__),
                  note='A negative calibration result is retained, not replaced by a best training seed. This record does not declare goal completion or a risk guarantee for uncertified profiles.')
    save(LAUNCH, record)
    print(json.dumps(record), flush=True)


def check_launch():
    require_committed([LAUNCH])
    record = json.loads(LAUNCH.read_text())
    if record['launcher_sha256'] != digest(__file__):
        raise ValueError('Final launcher changed after preparation')
    for path, expected in record['input_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen launch input changed: ' + path)
    inputs()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('check')
    p = sub.add_parser('worker')
    p.add_argument('shard', type=int)
    sub.add_parser('finalize')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        record = check_launch()
        if args.command == 'worker':
            worker(FINAL, args.shard)
        elif args.command == 'finalize':
            if (FINAL / 'summary.json').exists():
                raise FileExistsError('Preserve the finalized final-test artifact')
            finalize(FINAL)
        else:
            print(json.dumps(record), flush=True)
