"""Review complete calibration with the already-frozen independent risk auditor.

No final-confirmation outcomes are read. Calibration gains remain calibration
statistics and cannot satisfy the independent usefulness acceptance test.
"""
import json
from pathlib import Path
import shutil

import numpy as np

from audit_final_report_v19 import audit_risk, modes_for, observation, same
from compare_established_v19 import complete_study
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol

ROOT = Path(__file__).resolve().parents[1]
V19 = ROOT / 'artifacts/v19'


def run():
    output = V19 / 'calibration_review'
    if output.exists():
        raise FileExistsError('Preserve an existing completed-calibration review')
    study, calibration = V19 / 'risk_validation', V19 / 'comparison_calibration'
    require_committed([study / 'summary.json', study / 'sha256.json', calibration / 'calibration.json', calibration / 'conditional_risk.json'])
    read_protocol(study)
    verify_manifest(calibration)
    verify_manifest(V19 / 'final_report_plan')
    frozen = json.loads((V19 / 'final_report_plan/plan.json').read_text())
    if digest(ROOT / frozen['source_path']) != frozen['source_sha256']:
        raise ValueError('Frozen independent auditor source changed')
    protocol, lookup, names, _, _ = complete_study(study, 'risk_validation')
    spec = protocol['spec']
    risk = json.loads((calibration / 'conditional_risk.json').read_text())
    calibrated = json.loads((calibration / 'calibration.json').read_text())
    reconstructed = audit_risk(lookup, spec, risk)
    actual_modes = modes_for(spec, spec['risk_validation']['mode_selection_stream'])
    mean_records = {r['candidate']:r for r in risk['mean_improvement_comparator']['candidates']}
    table = []
    for name in names:
        paired = [observation(lookup[i,mode,name],lookup[i,mode,'baseline']) for i,mode in enumerate(actual_modes)]
        gain = float(np.mean([r['gain'] for r in paired]))
        same(gain,mean_records[name]['mean_gain'],name + ' independently reconstructed calibration gain')
        actual = reconstructed[name]['actual_random_mode']
        bundle = reconstructed[name]['worst_mode_bundle']
        table.append(dict(candidate=name, actual_adopted=actual['adopted'], actual_adoption_percent=100*actual['adopted']/spec['n_initials'],
                          actual_harmful=actual['harmful'], actual_upper_percent=100*actual['threshold_upper'],
                          bundle_adopted=bundle['adopted'], bundle_harmful=bundle['harmful'], bundle_upper_percent=100*bundle['threshold_upper'],
                          calibration_gain_seconds=gain, calibration_gain_percent=100*gain/np.mean([r['baseline_cost'] for r in paired]),
                          both_targets_certified=actual['certified'] and bundle['certified']))
    selected = {}
    for seed in sorted(protocol['models']):
        options = [name for name in names if name.endswith('_'+seed) and name in risk['both_targets_certified']]
        selected[seed] = max(options,key=lambda name:mean_records[name]['mean_gain']) if options else 'baseline'
    same(selected,calibrated['ltt_selection'],'independently reconstructed frozen LTT selection')
    report = dict(purpose='Independent complete-calibration review, not final usefulness validation',
                  independent_initials=spec['n_initials'], actual_sumo_runs=spec['expected_actual_sumo_runs'],
                  all_18_conditional_tests_reconstructed=True, reconstructed_risk=reconstructed, table=table,
                  both_targets_certified=risk['both_targets_certified'], all_three_strict_models_certified=all(r['both_targets_certified'] for r in table if r['candidate'].startswith('portfolio0_')),
                  ltt_selection=selected, mean_spi_selection=calibrated['mean_spi_selection'], conformal=calibrated['conformal'],
                  final_confirmation_outcomes_read=False, goal_completion_claimed=False,
                  scope=risk['scope'], source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in (Path(__file__).resolve(),ROOT/'tools/audit_final_report_v19.py')},
                  input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in (study/'episodes.json',study/'protocol.json',calibration/'conditional_risk.json',calibration/'calibration.json')})
    output.mkdir()
    for path in report['source_sha256']:
        destination=output/'source'/path
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/path,destination)
    save(output/'review.json',report)
    save(output/'sha256.json',{p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file() and p.name!='sha256.json'})
    print(json.dumps({'strict_gate_calibration':[r for r in table if r['candidate'].startswith('portfolio0_')],
                      'all_18_conditional_tests_reconstructed':True,'all_three_strict_models_certified':report['all_three_strict_models_certified']},indent=2),flush=True)


if __name__ == '__main__':
    run()
