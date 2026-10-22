"""Preserve every coupled actor-update dataset and model for reconstruction."""
import argparse,hashlib,json,shutil
from pathlib import Path
from coopgate.cooperative_iteration_v16 import load_coupled


def archive(source,destination):
    source,destination=Path(source),Path(destination)
    _,meta=load_coupled(source/'policy.pt')
    if meta['updates']!=meta['target_updates']:raise ValueError('Incomplete fixed-budget run')
    destination.mkdir(parents=True,exist_ok=False)
    for path in source.iterdir():
        if path.is_dir():shutil.copytree(path,destination/path.name)
        else:shutil.copy2(path,destination/path.name)
    log=source.with_suffix('.log')
    if log.exists():shutil.copy2(log,destination/'console.txt')
    manifest={str(p.relative_to(destination)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.rglob('*')) if p.is_file()}
    (destination/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(destination=str(destination),assets=len(manifest),updates=meta['updates'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination')
    a=p.parse_args();archive(a.source,a.destination)
