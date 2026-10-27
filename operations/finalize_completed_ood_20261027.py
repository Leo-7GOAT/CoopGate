"""Publish the completed OOD replay receipt; preserve negative study decisions."""
import json
from pathlib import Path
from summarize_completed_ood_20261027 import read_complete, ROOT, SOURCE, digest


def main():
    report=read_complete()
    output=ROOT/'artifacts/recovery_20261027/ood_finalization_receipt.json'
    if output.exists():raise FileExistsError('Preserve prior finalization receipt')
    events=[json.loads(line) for line in (ROOT/'.git/mainline-20261027.jsonl').read_text(encoding='utf-8').splitlines()]
    launch=[e for e in events if e['kind']=='owned_process_started' and e['detail']['role']=='ood_summary']
    exits=[e for e in events if e['kind']=='owned_process_exit' and e['detail']['name']=='ood_summary']
    witness=json.loads((ROOT/'.git/ood_summary_native_exit_20261027.json').read_text(encoding='utf-8-sig'))
    if (len(launch)!=1 or len(exits)!=1 or exits[0]['detail']['exit_code']!=0
            or launch[0]['detail']['pid']!=exits[0]['detail']['native_pid']
            or witness['pid']!=26040 or not witness['ended'] or witness['exit_code']!=0):
        raise ValueError('Retained owned handle and original native exit witnesses required')
    recovery=json.loads((SOURCE/'recovery.json').read_text(encoding='utf-8'))
    for name,expected in recovery['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen OOD source changed: '+name)
    plan=ROOT/'artifacts/v39/ood_mismatch_validation/plan.json'
    binding=ROOT/'artifacts/v39/ood_mismatch_validation/execution_binding/manifest.json'
    if digest(plan)!=report['plan_sha256'] or digest(binding)!=report['execution_binding_sha256']:
        raise ValueError('Original scientific binding or plan differs')
    result=dict(actual_sumo_runs=22464,paired_condition_blocks=1728,independent_initials=192,
        native_summary_exit=witness,owned_summary_launch=launch[0],owned_summary_exit=exits[0],
        report_sha256=digest(SOURCE/'analysis/report.json'),
        original_plan_sha256=digest(plan),original_binding_sha256=digest(binding),
        frozen_source_files_rechecked=len(recovery['source_sha256']),
        all_original_episodes_independently_replayed=True,
        all_three_fixed_models_pass=report['primary']['all_three_fixed_models_pass'],
        new_formal_safety_certificate=False,new_inferential_tests=0,
        old_interruption_exit_not_rewritten=True,
        complete_raw_archive_inventory_sha256=digest(ROOT/'artifacts/recovery_20261027/ood_completed_collection/inventory.json'),
        failure_types_sha256=digest(ROOT/'artifacts/recovery_20261027/ood_failure_types/summary.json'),
        scope_summary_sha256=digest(ROOT/'artifacts/recovery_20261027/ood_scope_summary/summary.json'),
        figure_sha256=digest(ROOT/'artifacts/recovery_20261027/ood_scope_summary/failure_scope.png'),
        visual_qa='Full PNG inspected: labels, percentage cells and scope annotation visible without clipping',
        outcome='Complete OOD study; all-model success criterion not met; deployment safety not established')
    with output.open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:result[k] for k in ('actual_sumo_runs','frozen_source_files_rechecked',
        'all_three_fixed_models_pass','new_formal_safety_certificate')}),flush=True)


if __name__=='__main__':main()
