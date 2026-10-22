"""Freeze every fresh model and evaluation source before the held-out SUMO test."""
import hashlib,json,shutil,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/v14/single'


def freeze():
    if (D/'protocol.json').exists():raise RuntimeError('Test protocol already frozen')
    training_path=D/'training_protocol.json';p=json.loads(training_path.read_text())
    for name,digest in p['training_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    roots={s:ROOT/f'runs/v14_single_release/seed_{s}' for s in p['training_seeds']}
    for seed,root in roots.items():
        record=json.loads((root/'complete.json').read_text())
        assert record['protocol_sha256']==hashlib.sha256(training_path.read_bytes()).hexdigest()
        assert record['final_policy_sha256']==hashlib.sha256((root/'averaged/policy.pt').read_bytes()).hexdigest()
        for phase in ('phase_a','phase_b'):
            meta=json.loads((root/phase/'metadata.json').read_text());config=p[phase]
            assert meta['iteration']==meta['iterations']==config['iterations']
            assert meta['visited_states']==config['iterations']*config['states']
            assert meta['epochs']==config['epochs'] and meta['width']==p['width']
    for seed,root in roots.items():
        target=D/f'seed_{seed}';target.mkdir(exist_ok=False)
        for phase in ('phase_a','phase_b','averaged'):
            dst=target/phase;dst.mkdir()
            for path in (root/phase).iterdir():
                if path.is_file() and path.suffix in ('.pt','.json','.csv'):shutil.copy2(path,dst/path.name)
        for name in ('complete.json','launch.json'):shutil.copy2(root/name,target/name)
        shutil.copy2(root/'phase_a/initial.pt',target/'initial.pt')
        shutil.copy2(root/'phase_b/policy.pt',target/'last.pt')
        shutil.copy2(root/'averaged/policy.pt',target/'policy.pt')
    sources=list(p['training_source_sha256'])+['tools/freeze_single_v14.py','tools/study_single_v14.py',
        'tools/report_single_v14.py','tools/verify_single_v14.py','tools/experiment_ego_v13.py','tools/verify_single_v12.py',
        'tools/develop_v12.py','tools/archive_single_v14.py']
    sources += [str(file.relative_to(ROOT)).replace('\\','/') for file in D.rglob('*') if file.is_file()]
    methods=['rollout']+p['primary_methods']+p['secondary_methods']
    protocol=dict(version='v14-single-test',training_seeds=p['training_seeds'],
        training_protocol_sha256=hashlib.sha256(training_path.read_bytes()).hexdigest(),
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        initials=p['test_initials'],disturbances=p['disturbances'],methods=methods,
        primary_methods=p['primary_methods'],secondary_methods=p['secondary_methods'],
        expected_runs=len(p['test_initials'])*len(p['disturbances'])*len(methods),
        criteria=p['criteria'],bootstrap=p['bootstrap'],
        audit_initial_indices=[0,1,128,129,256,257,384,385],replay_initial_indices=[0,1],
        input_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(set(sources))})
    (D/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(expected_runs=protocol['expected_runs'],frozen_inputs=len(protocol['input_sha256']),
                         training_seeds=p['training_seeds'])),flush=True)


if __name__=='__main__':freeze()
