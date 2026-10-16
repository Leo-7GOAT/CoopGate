"""Archive a completed SUMO mechanism study and its matching replay audit."""
import argparse,hashlib,json,shutil
from pathlib import Path


def archive(study,audit,destination):
    study,audit,destination=Path(study),Path(audit),Path(destination)
    p=json.loads((study/'protocol.json').read_text())
    rows=json.loads((study/'episodes.json').read_text());r=json.loads(audit.read_text())
    assert len(rows)==p['expected_sumo_runs']==r['actual_sumo_runs']
    assert hashlib.sha256((study/'episodes.json').read_bytes()).hexdigest()==r['episodes_sha256']
    assert r['additional_sumo_replays']>0
    destination.mkdir(parents=True,exist_ok=False);shutil.copytree(study,destination/'study')
    shutil.copy2(audit,destination/'audit.json')
    log=study.with_suffix('.log')
    if log.exists():shutil.copy2(log,destination/'console.txt')
    source=Path(__file__).parent
    for name in ('verify_composed_gate_v18.py','verify_advantage_gate_v18.py',Path(__file__).name):
        target=destination/'audit_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    hashes={str(f.relative_to(destination)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(destination.rglob('*')) if f.is_file()}
    (destination/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(destination=str(destination),assets=len(hashes))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('audit');p.add_argument('destination')
    a=p.parse_args();archive(a.study,a.audit,a.destination)
