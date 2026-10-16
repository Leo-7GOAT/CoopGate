"""Preserve development outcomes, including numerical failures, byte for byte."""
import argparse, hashlib, json, shutil
from pathlib import Path


def archive(source, destination, status, explanation):
    source, destination = Path(source), Path(destination)
    if not source.is_dir() or status not in ('failed','complete'):
        raise ValueError('Existing attempt and explicit outcome required')
    if status == 'complete' and not (source/'metadata.json').exists():
        raise ValueError('Missing terminal training metadata')
    destination.mkdir(parents=True,exist_ok=False)
    # Intermediate budget checkpoints are retained in runs; the last state and
    # complete training history are sufficient to identify a failed attempt.
    names=['protocol.json','initial.pt','policy.pt','training.csv','metadata.json','episodes.json',
           'failure.json','failure_batch.pt','source']
    for name in names:
        path=source/name
        if path.is_dir():shutil.copytree(path,destination/name)
        elif path.is_file():shutil.copy2(path,destination/name)
    log=source.with_suffix('.log')
    if log.is_file():shutil.copy2(log,destination/'console.log')
    (destination/'outcome.json').write_text(json.dumps(dict(status=status,explanation=explanation,
        source=str(source),checkpoint_interpretation='Last saved state; only complete fixed-budget runs are eligible'),indent=2)+'\n')
    manifest={str(p.relative_to(destination)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.rglob('*')) if p.is_file()}
    (destination/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(destination=str(destination),assets=len(manifest),status=status)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination')
    p.add_argument('--status',choices=['failed','complete'],required=True);p.add_argument('--explanation',required=True)
    a=p.parse_args();archive(a.source,a.destination,a.status,a.explanation)
