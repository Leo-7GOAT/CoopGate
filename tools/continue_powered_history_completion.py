"""Own the fixed post-primary obligations without selecting on interim effects.

Wait for the primary owner's actual native terminal witness. A negative but
complete scientific decision still triggers BOTH previously registered paired
ablations. Native execution failure stops, preserving every original output.
This scheduler changes no controller, registration, sample or scientific rule.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import mappo_powered_history_formal_experiment as e
from mappo_completed_powered_attempt5 import complete_evidence
from mappo_native_launch import command

OUT = e.BASE / 'powered_history_completion_pipeline_20261103'


def now():
    return datetime.now(timezone.utc).isoformat()


def primary_ready():
    path = e.FORMAL / 'actual_root_native_terminal.json'
    if not path.exists():
        return False
    terminal = e.read(path)
    launch = e.read(e.FORMAL / 'owner_launch.json')
    if (terminal['native_pid'] != launch['native_pid']
            or not terminal['observed_by_direct_native_wait']
            or terminal['plan_sha256'] != e.sha(e.FORMAL / 'plan.json')):
        raise ValueError('Wrong actual primary native terminal witness')
    if terminal['actual_native_exit_code'] != 0:
        raise RuntimeError('Primary native execution failed; retain all outputs')
    # Owner writes terminal before the success receipt; wait for its atomic write.
    if not (e.FORMAL / 'orchestrator_native_receipt.json').exists():
        return False
    return True


def owned_child(label, script, args, report, owner_path):
    if owner_path.exists():
        raise FileExistsError('Never replace an existing native receipt')
    started = now()
    argv = command(script, *args)
    with (OUT / (label + '.log')).open('xb') as stream:
        child = subprocess.Popen(argv, cwd=e.ROOT, stdout=stream, stderr=subprocess.STDOUT)
        e.write(OUT / (label + '_started.json'), dict(
            native_pid=child.pid, command=argv, started_utc=started))
        code = child.wait()
    receipt = dict(native_pid=child.pid, actual_native_exit_code=code,
                   observed_by_direct_native_wait=True, started_utc=started,
                   finished_utc=now(), scheduler_native_pid=os.getpid())
    if code == 0:
        receipt['report_sha256'] = e.sha(report)
    e.write(owner_path, receipt)
    e.write(OUT / (label + '_native_exit.json'), receipt)
    print(json.dumps(dict(stage=label, **receipt)), flush=True)
    if code:
        raise RuntimeError('Retain failed/incomplete child; do not replace observations: ' + label)
    return receipt


def run():
    OUT.mkdir(exist_ok=False)
    e.write(OUT / 'launch.json', dict(native_pid=os.getpid(), started_utc=now(),
        source_sha256=e.sha(__file__), primary_plan_sha256=e.sha(e.FORMAL / 'plan.json'),
        fixed_order=['compact', 'phase', 'diagnosis', 'figures'],
        completion_required_regardless_of_scientific_verdict=True,
        primary_effects_read_at_launch=False, new_error_allocated=False))
    while not primary_ready():
        print(json.dumps(dict(state='waiting_for_actual_primary_terminal', utc=now())), flush=True)
        time.sleep(30)
    cert, _ = complete_evidence()
    # Do not branch on scientific success: both registered variants always run.
    print(json.dumps(dict(state='primary_native_complete',
        joint_deployment_certified=cert['joint_deployment_certified'])), flush=True)
    receipts = {}
    for variant in ('compact', 'phase'):
        directory = e.BASE / ('powered_history_' + variant + '_holdout_ablation_20261103')
        if (directory / 'execution').exists() or (directory / 'report.json').exists():
            raise FileExistsError('Do not duplicate a previously launched ablation')
        receipts[variant] = owned_child(variant,
            e.ROOT / 'tools/run_powered_history_holdout_ablation.py',
            ['run', '--variant', variant], directory / 'report.json',
            directory / 'owner_native_exit.json')
    diagnosis = e.BASE / 'formal_mappo_attempt5_powered_diagnosis_20261103'
    receipts['diagnosis'] = owned_child('diagnosis',
        e.ROOT / 'tools/diagnose_mappo_powered_history_outcomes.py', [],
        diagnosis / 'report.json', OUT / 'diagnosis_owner_native_exit.json')
    figures = e.BASE / 'formal_mappo_attempt5_powered_figures_20261103'
    receipts['figures'] = owned_child('figures',
        e.ROOT / 'tools/plot_mappo_powered_history_results.py', ['--out', str(figures)],
        figures / 'report.json', OUT / 'figures_owner_native_exit.json')
    result = dict(completed=True, primary_actual_sumo_executions=45056,
        supplementary_actual_sumo_executions=6144, independent_initials=5120,
        new_error_allocated=False, joint_deployment_certified=cert['joint_deployment_certified'],
        primary_certificate_sha256=e.sha(e.FORMAL / 'analysis/certificate.json'),
        children=receipts, manuscript_update_and_visual_review_still_required=True,
        full_final_publication_still_required=True, finished_utc=now())
    e.write(OUT / 'report.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        # This only inspects the actual native witness, never partial outcomes.
        print(json.dumps(dict(primary_native_ready=primary_ready(),
                              pipeline_already_launched=OUT.exists())), flush=True)
    else:
        run()
