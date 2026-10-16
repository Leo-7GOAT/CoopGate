"""Preserve all completed V14 development attempts, including negative results."""
import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/v14/development'
TRAINING=('v14_pi_pilot_801','v14_pi_rule_801','v14_pi_large_803','v14_certificate_pilot_805',
          'v14_certificate_large_807','v14_kinematic_pilot_809','v14_pi_fresh_targets_813')
VALIDATION=('v14_validation_stage1','v14_validation_stage2','v14_validation_stage3','v14_validation_average')


def archive():
    if D.exists():raise RuntimeError('Development archive already exists')
    for name in TRAINING:
        root=ROOT/'runs'/name;meta=json.loads((root/'metadata.json').read_text())
        assert meta['iteration']==meta['iterations']
        for file,digest in meta['source_sha256'].items():
            assert hashlib.sha256((root/'source'/file).read_bytes()).hexdigest()==digest,(name,file)
    for name in VALIDATION:assert (ROOT/'runs'/name/'summary.json').is_file()
    D.mkdir(parents=True);records=[]
    for name in (*TRAINING,'v14_pi_average_813'):
        source=ROOT/'runs'/name;target=D/name;target.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.suffix in ('.pt','.json','.csv'):shutil.copy2(path,target/path.name)
        if (source/'source').exists():shutil.copytree(source/'source',target/'source')
        meta=json.loads((source/'metadata.json').read_text())
        samples={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source.glob('data_*.npz')}
        records.append(dict(name=name,kind='training' if name in TRAINING else 'parameter_averaging_only',
            visited_states=meta['visited_states'] if name in TRAINING else 0,
            evaluated_model_transitions=meta['evaluated_model_transitions'] if name in TRAINING else 0,
            local_training_sample_sha256=samples))
    for name in VALIDATION:shutil.copytree(ROOT/'runs'/name,D/name)
    reports=D/'reports';reports.mkdir()
    for path in (ROOT/'runs').glob('v14_*.json'):shutil.copy2(path,reports/path.name)
    manifest=dict(purpose='Development and method selection only; not the frozen three-seed single-ego test',
        training_runs=records,visited_training_states=sum(r['visited_states'] for r in records),
        evaluated_model_transitions=sum(r['evaluated_model_transitions'] for r in records),
        actual_sumo_runs=sum(json.loads((D/name/'summary.json').read_text())['sumo_runs'] for name in VALIDATION),
        independent_validation_initials=sum(json.loads((D/name/'protocol.json').read_text())['initials'] for name in VALIDATION),
        notes=['All fixed final models and all intermediate checkpoints retained, including regressions',
               'Sources with matching training-protocol hashes are bundled for changed development versions',
               'Training sample archives remain local and are hash-listed; source and seeds permit regeneration',
               'Exact-model diagnostics are not counted as SUMO executions',
               'The averaged policy reuses one training pipeline and is not an additional training seed or MARL',
               'No single-ego formal prerequisite verdict is issued from these development data'])
    (D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    hashes={str(p.relative_to(D)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*') if p.is_file()}
    (D/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='training_runs'}),flush=True)


if __name__=='__main__':archive()
