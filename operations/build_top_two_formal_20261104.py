"""Generate exclusive prospective adapters; never edit an existing study."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/mappo_ctde_20261101'
RENAME = {
    'mappo_powered_history': 'mappo_top_two_formal',
    'mappo_powered_risk_attempt5': 'mappo_top_two_risk_attempt6',
    'formal_mappo_attempt5_powered_20261103': 'formal_mappo_attempt6_top_two_20261104',
    'powered_history_formal_adapter_validation_20261103': 'top_two_formal_adapter_validation_20261104',
    'mappo_formal_attempt5_powered_20261103_protocol.md': 'mappo_top_two_formal_attempt6_20261104_protocol.md',
    'observed_history_fixed_candidate_20261103.json': 'top_two_fixed_candidate_20261104.json',
    'history_power_design_20261103/report.json': 'top_two_formal_power_20261104/report.json',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(text):
    return {n.name: ast.dump(ast.Module(body=n.body, type_ignores=[]))
            for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def replace(text):
    for a, b in sorted(RENAME.items(), key=lambda v: -len(v[0])):
        text = text.replace(a, b)
    return text


def run():
    power = BASE / 'top_two_formal_power_20261104'
    pr = json.loads((power / 'report.json').read_bytes())
    owner = json.loads((power / 'owner_native_exit.json').read_bytes())
    if (owner['actual_native_exit_code'] != 0 or owner['report_sha256'] != sha(power / 'report.json')
            or pr['selected_design'] != dict(calibration_initials=6144, confirmation_initials=1024, actual_runs_if_registered=59392)):
        raise ValueError('Complete fixed sizing rule must select 6144/1024')
    outputs = {}
    proof = []
    names = ['mappo_powered_risk_attempt5', 'mappo_powered_history_joint_data',
             'mappo_powered_history_formal_design', 'mappo_powered_history_formal_experiment',
             'mappo_powered_history_formal_finalize', 'run_mappo_powered_history_formal_validation']
    for name in names:
        source = ROOT / 'tools' / (name + '.py')
        original = source.read_text(encoding='utf-8')
        text = replace(original)
        # The prefix replacement intentionally creates shorter, unique module names.
        text = text.replace('mappo_top_two_formal_formal', 'mappo_top_two_formal')
        text = text.replace('mappo_top_two_formal_joint_data', 'mappo_top_two_joint_data')
        if name == 'mappo_powered_risk_attempt5':
            text = text.replace('CALIBRATION_N = 4096', 'CALIBRATION_N = 6144').replace('ATTEMPT = 5', 'ATTEMPT = 6')
            text = text.replace('attempt5 adapter', 'attempt6 adapter')
        if name.endswith('_design'):
            text = text.replace('calibration=4096', 'calibration=6144').replace('{p:1024 for p in PLANTS}', '{p:1536 for p in PLANTS}')
            for a, b in [('875000001', '900000001'), ('876000001', '901000001'), ('880000001', '902000001'),
                         ('550101', '650101'), ('550201', '650201'), ('550301', '650301'), ('mappo5:', 'mappo6:')]:
                text = text.replace(a, b)
            text = text.replace('candidate["controller"]!="ObservedHistoryGate"', 'candidate["controller"]!="TopTwoHistoryGate"')
        if name.endswith(('_experiment', '_finalize')):
            text = text.replace('20480', '28672').replace('45056', '59392').replace('independent_initials=5120', 'independent_initials=7168')
            text = text.replace('calibration_initials=4096', 'calibration_initials=6144').replace('n=4096;lookup', 'n=6144;lookup')
        if name.endswith('_experiment'):
            start = text.index('def source_files():')
            end = text.index('\n\ndef build(', start)
            sf = '''def source_files():
    pilot=read(BASE/'development/local_top_two_fresh_pilot_20261104/plan.json')
    paths=set(pilot['sources'])
    paths.update([
        'tools/mappo_powered_history_formal_runtime.py',
        'tools/mappo_powered_history_joint_data.py','tools/mappo_powered_risk_attempt5.py',
        'tools/mappo_top_two_risk_attempt6.py','tools/mappo_top_two_joint_data.py',
        'tools/mappo_top_two_formal_runtime.py','tools/mappo_top_two_formal_design.py',
        'tools/mappo_top_two_formal_experiment.py','tools/mappo_top_two_formal_finalize.py',
        'tools/run_mappo_top_two_formal_validation.py','tools/run_mappo_top_two_formal.py',
        'tools/own_mappo_top_two_formal.py','tools/audit_mappo_attempt6_predecessor.py',
        'tools/mappo_completed_powered_attempt5.py','tools/mappo_powered_history_formal_experiment.py',
        'tools/mappo_powered_history_formal_design.py',
        'operations/build_top_two_formal_20261104.py','operations/top_two_formal_power_20261104.py',
        'docs/mappo_top_two_formal_attempt6_20261104_protocol.md',
        'docs/top_two_formal_power_20261104_protocol.md',
        'tests/test_mappo_top_two_formal_adapters.py','tests/test_mappo_top_two_formal_integrity.py',
        'tests/test_mappo_top_two_formal_failures.py','tests/test_top_two_formal_power.py',
        'tests/test_mappo_top_two_same_pool.py'])
    return [ROOT/name for name in sorted(paths)]
'''
            text = text[:start] + sf + text[end:]
            text = text.replace("allocation['attempt']!=5 or allocation['allocated_error']!=.05/30", "allocation['attempt']!=6 or allocation['allocated_error']!=.05/42")
        ast.parse(text)
        if name in ('mappo_powered_risk_attempt5', 'mappo_powered_history_joint_data') and functions(text) != functions(original):
            raise ValueError('Original statistical or paired accounting function bodies changed')
        target_name = replace(name).replace('mappo_top_two_formal_formal', 'mappo_top_two_formal').replace('mappo_top_two_formal_joint_data', 'mappo_top_two_joint_data')
        target = ROOT / 'tools' / (target_name + '.py')
        outputs[target] = text
        proof.append(dict(source=source.relative_to(ROOT).as_posix(), source_sha256=sha(source), target=target.relative_to(ROOT).as_posix()))
    for suffix in ('adapters', 'integrity', 'failures'):
        source = ROOT / 'tests' / ('test_mappo_powered_history_formal_' + suffix + '.py')
        text = replace(source.read_text()).replace('mappo_top_two_formal_formal', 'mappo_top_two_formal').replace('mappo_top_two_formal_joint_data', 'mappo_top_two_joint_data')
        if suffix == 'adapters':
            text = text.replace('stats.ATTEMPT==5', 'stats.ATTEMPT==6').replace('.05/30', '.05/42')
            text = text.replace('len(cal)==4096', 'len(cal)==6144').replace('dict.fromkeys(design.PLANTS,1024)', 'dict.fromkeys(design.PLANTS,1536)')
            text = text.replace('==5120', '==7168').replace('==45056', '==59392').replace('==20480', '==28672')
            text = text.replace('((4096,3)', '((6144,3)').replace('design.PLANTS*1024', 'design.PLANTS*1536').replace('np.zeros(4096,bool)', 'np.zeros(6144,bool)')
        if suffix == 'failures':
            # Original-field equivalence is now intentionally replaced by a
            # same-pool equivalence test; retaining that old assertion would
            # incorrectly demand removing the newly registered candidate pool.
            node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'test_nominal_control_preserves_all_original_fields_without_false_bank_label')
            lines = text.splitlines(keepends=True)
            begin = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
            text = ''.join(lines[:begin] + lines[node.end_lineno:])
        ast.parse(text)
        outputs[ROOT / 'tests' / ('test_mappo_top_two_formal_' + suffix + '.py')] = text
    for path in outputs:
        if path.exists():
            raise FileExistsError('Preserve existing adapter: ' + str(path))
    old = json.loads((BASE / 'observed_history_fixed_candidate_20261103.json').read_bytes())
    pilot = json.loads((BASE / 'development/local_top_two_fresh_pilot_20261104/plan.json').read_bytes())
    candidate = {k: old[k] for k in ('common_gain_floor', 'actor_seeds', 'actor_identities', 'algorithm', 'initialization', 'runtime',
        'required_mixture_conditional_harm_upper', 'required_strong_lag_conditional_harm_upper', 'required_adoption_lower',
        'required_mean_paired_percentage_gain_t_lower_strictly_above', 'required_observed_baseline_or_candidate_failures',
        'same_adoption_controls_required', 'coupled_bank_sha256')}
    candidate.update(controller='TopTwoHistoryGate', controller_source_sha256=pilot['sources'],
        formal_calibration_initials=6144, formal_confirmation_initials=1024,
        selection='Post-pilot fixed family, all three seeds; earlier reused-data screen remains failed; fresh pilot is development only.',
        development_evidence_sha256=sha(BASE / 'development/local_top_two_fresh_analysis_20261104/report.json'),
        sizing_evidence_sha256=sha(power / 'report.json'), error_allocation_attempt_reserved=None,
        formal_experiment_started=False, risk_certificate=False, paper_ready=False,
        raw_MAPPO_or_central_critic_causal_superiority_proved=False)
    candidate_path = BASE / 'top_two_fixed_candidate_20261104.json'
    if candidate_path.exists():
        raise FileExistsError('Preserve candidate')
    for path, text in outputs.items():
        with path.open('x', encoding='utf-8', newline='\n') as f:
            f.write(text)
    with candidate_path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(candidate, f, indent=2); f.write('\n')
    for item in proof:
        item['target_sha256'] = sha(ROOT / item['target'])
    out = BASE / 'top_two_adapter_generation_20261104.json'
    with out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(dict(adapters=proof, original_statistics_and_paired_function_bodies_identical=True,
            online_controller_changed_to_fixed_top_two=True, nominal_comparator_receives_same_candidate_provider=True,
            original_physics_execution_reused_unchanged=True, candidate_sha256=sha(candidate_path),
            new_error_allocated=False, new_formal_runs=0, generator_sha256=sha(__file__)), f, indent=2); f.write('\n')
    print(json.dumps(dict(adapters=len(outputs), calibration=6144, confirmation=1024, actual_runs_if_registered=59392, new_error_allocated=False)))


if __name__ == '__main__':
    run()
