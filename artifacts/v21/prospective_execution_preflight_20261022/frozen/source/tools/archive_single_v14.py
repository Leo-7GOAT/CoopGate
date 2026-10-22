"""Archive the complete preregistered test, including a negative verdict."""
import argparse,hashlib,json,shutil
from pathlib import Path
from report_single_v14 import summarize

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/v14/single'


def archive(study):
    study=Path(study);protocol_path=D/'protocol.json'
    protocol=json.loads(protocol_path.read_text())
    if (D/'study').exists():raise RuntimeError('Frozen test already archived')
    for name,digest in protocol['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    manifest=json.loads((study/'manifest.json').read_text())
    assert manifest['protocol_sha256']==hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert manifest['sumo_runs']==protocol['expected_runs']
    rows=json.loads((study/'episodes.json').read_text());summary=summarize(protocol,rows)
    for index in protocol['audit_initial_indices']:
        for mode in protocol['disturbances']:
            for method in protocol['methods']:
                assert (study/f'trace_{index}_{mode}_{method}.json').is_file()
    shutil.copytree(study,D/'study')
    (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    hashes={str(path.relative_to(D)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(D.rglob('*')) if path.is_file() and path.name!='sha256.json'}
    (D/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(executions=len(rows),acceptance=summary['acceptance'],
                         gain_percent=summary['gain_percent'],assets=len(hashes))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',default='runs/v14_single_study');a=p.parse_args();archive(a.study)
