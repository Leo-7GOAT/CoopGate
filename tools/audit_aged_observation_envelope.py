"""Post-hoc public-input envelope audit on ALL completed V45 native traces.

No running-study outcomes, new SUMO, fitting, control changes or sample selection.
Estimates are all formed before any true state/memory is read for their audit.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from coopgate.aged_observation_envelope import PublicEnvelope, estimate_frames
from coopgate.episode_store_v21 import inventory, observations

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'artifacts/v45/bounded_observation_development'
OUT = ROOT/'artifacts/diagnostics/aged_observation_envelope'
SOURCES = ('src/coopgate/aged_observation_envelope.py',
    'tools/audit_aged_observation_envelope.py', 'tests/test_aged_observation_envelope.py',
    'docs/aged_observation_envelope.md', 'src/coopgate/episode_store_v21.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def committed(path):
    name = Path(path).relative_to(ROOT).as_posix()
    if subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT) != Path(path).read_bytes():
        raise ValueError('Commit unchanged input first: '+name)


def public_frames(trace, terminal):
    frames = []
    previous = np.zeros(3)
    for entry in trace:
        frames.append(dict(observed_state=entry['observed_state'], perception=entry['perception'],
                           previous_command=previous.tolist()))
        previous = np.asarray(entry['command'])
    if terminal is not None:
        np.testing.assert_array_equal(previous, terminal['previous_command'])
        frames.append(dict(observed_state=terminal['observed_state'], perception=terminal['perception'],
                           previous_command=previous.tolist()))
    return frames


def audit():
    if OUT.exists():
        raise FileExistsError('Preserve the first envelope diagnosis')
    manifest = read(PARENT/'manifest.json'); committed(PARENT/'manifest.json')
    assert manifest['test_exit_code']==0 and manifest['actual_sumo_runs']==39
    expected = {f'{i:05d}/{name}' for i in range(3) for name in
        ('complete.json','development_group.json','development_plan.json','observations.sqlite',
         'observations.sqlite.seal.json')}
    assert set(manifest['files_sha256']) == expected
    for name, sha in manifest['files_sha256'].items():
        assert digest(PARENT/name)==sha
    for name in SOURCES:
        committed(ROOT/name)
    sources = {name:digest(ROOT/name) for name in SOURCES}
    b = PublicEnvelope()
    # Phase one consumes only explicitly selected public inputs.
    prepared = []
    for index in range(3):
        directory = PARENT/f'{index:05d}'
        group = read(directory/'development_group.json')
        meta = dict(block_index=index, execution_binding_sha256='same-frozen-v39-runtime',
            initial_id=group['initial_id'], plan_sha256='development-projection-v45',
            stage='v39_ood_mismatch_validation')
        identities = inventory(directory/'observations.sqlite')['cases']
        records = list(observations(directory/'observations.sqlite', meta, identities))
        assert len(records)==13
        for key, body, trace in records:
            assert key[0]==index and key[5]==group['perception']
            terminal = body['row']['terminal_attempt']
            frames = public_frames(trace, terminal)
            estimates = estimate_frames(frames, group['perception'])
            prepared.append((key,body,trace,terminal,frames,estimates))
    assert len(prepared)==39
    # Only after every estimate is fixed is truth opened for checking coverage.
    rows = []
    for key,body,trace,terminal,frames,estimates in prepared:
        plant = body['row']['plant']
        assert np.all((np.asarray(plant['gain'])>=b.gain_lower)&(np.asarray(plant['gain'])<=b.gain_upper))
        assert np.all((np.asarray(plant['quadratic_drag'])>=b.drag_lower)&(np.asarray(plant['quadratic_drag'])<=b.drag_upper))
        assert np.all((np.asarray(plant['lag_tau_seconds'])>=b.tau_lower)&(np.asarray(plant['lag_tau_seconds'])<=b.tau_upper))
        assert np.all((np.asarray(plant['command_delay_decisions'])>=1)&(np.asarray(plant['command_delay_decisions'])<=b.maximum_delay))
        truth = [entry['true_state_before'] for entry in trace]
        if terminal is not None:
            truth.append(terminal['true_state'])
        assert len(truth)==len(estimates)
        for step,(frame,estimate,true) in enumerate(zip(frames,estimates,truth)):
            true = np.asarray(true)
            memory = np.zeros(3) if step==0 else np.asarray(trace[step-1]['actuator_states'][-1])
            lo,hi = np.asarray(estimate['lower']),np.asarray(estimate['upper'])
            ml,mh = np.asarray(estimate['memory_lower']),np.asarray(estimate['memory_upper'])
            violation = float(max(0.,np.max(lo-true),np.max(true-hi)))
            memory_violation = float(max(0.,np.max(ml-memory),np.max(memory-mh)))
            assert violation <= 2e-8 and memory_violation <= 2e-10
            raw = np.asarray(frame['observed_state'])
            rows.append(dict(case=list(key),decision=step,terminal_attempt=step==len(trace),
                estimate=estimate,true_state=true.tolist(),true_memory=memory.tolist(),
                max_state_containment_violation=violation,max_memory_containment_violation=memory_violation,
                raw_absolute_error=np.abs(raw-true).tolist(),
                midpoint_absolute_error=np.abs((lo+hi)/2-true).tolist(),width=(hi-lo).tolist()))
    aggregate = {}
    for profile in ('clean_reference','bounded_bias_jitter','stale_dropout'):
        selected = [row for row in rows if row['case'][5]==profile]
        raw = np.asarray([r['raw_absolute_error'] for r in selected])
        midpoint = np.asarray([r['midpoint_absolute_error'] for r in selected])
        widths = np.asarray([r['width'] for r in selected])
        aggregate[profile] = dict(episodes=13,received_frames=len(selected),
            terminal_frames=sum(r['terminal_attempt'] for r in selected),
            maximum_age_decisions=max(max(r['estimate']['age_decisions']) for r in selected),
            maximum_state_violation=max(r['max_state_containment_violation'] for r in selected),
            maximum_memory_violation=max(r['max_memory_containment_violation'] for r in selected),
            raw_mean_absolute_position_error=float(raw[:,:,0].mean()),
            midpoint_mean_absolute_position_error=float(midpoint[:,:,0].mean()),
            raw_max_absolute_position_error=float(raw[:,:,0].max()),
            midpoint_max_absolute_position_error=float(midpoint[:,:,0].max()),
            maximum_position_interval_width=float(widths[:,:,0].max()),
            maximum_speed_interval_width=float(widths[:,:,1].max()))
    assert len(rows)==778+13
    assert {name:digest(ROOT/name) for name in SOURCES}==sources
    report = dict(parent_manifest_sha256=digest(PARENT/'manifest.json'),source_sha256=sources,
        public_model_envelope=asdict(b),aggregate=aggregate,rows=rows,
        estimates_fixed_before_truth_audit=True,all_39_completed_source_records_used=True,
        true_state_memory_parameters_and_future_innovation_used_by_estimator=False,
        public_information_enlarged_vs_original_controller=True,
        posthoc_development_diagnosis=True,reused_independent_initials=3,new_independent_initials=0,
        additional_sumo_executions=0,controls_or_formal_studies_changed=False,
        new_risk_certificate=False,original_physical_certificate_restored=False,
        numerical_interval_arithmetic_verified=False,performance_improvement_established=False)
    OUT.mkdir(parents=True,exist_ok=False)
    with (OUT/'report.json').open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(report,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(report_sha256=digest(OUT/'report.json'),received_frames=len(rows),
                         aggregate=aggregate,additional_sumo_executions=0)),flush=True)


if __name__=='__main__':
    audit()
