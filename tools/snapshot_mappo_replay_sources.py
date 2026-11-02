"""Preserve exact frozen replay-source bytes, including line endings."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def snapshot(directory):
    directory=Path(directory).resolve()
    scope=(ROOT/'artifacts/mappo_ctde_20261101/development').resolve()
    formal=(ROOT/'artifacts/mappo_ctde_20261101/formal_mappo_attempt3_20261102').resolve()
    if directory.parent!=scope and directory!=formal:raise ValueError('Explicit authorized MAPPO study directory required')
    if directory==formal:
        receipt=json.loads((directory/'completion_receipt.json').read_bytes())
        native=json.loads((directory/'orchestrator_native_receipt.json').read_bytes())
        if native['actual_native_exit_code']!=0 or receipt['actual_native_finalizer']['actual_native_exit_code']!=0:
            raise ValueError('Actual formal owner/finalizer terminal zero required')
    plan=json.loads((directory/'plan.json').read_text())
    target=directory/'source';target.mkdir(exist_ok=False)
    inventory=[]
    for name,expected in plan.get('source_sha256',plan.get('sources_sha256',{})).items():
        original=(ROOT/name).resolve()
        if not original.is_relative_to(ROOT) or not name.startswith(('src/','tools/','docs/','tests/')):
            raise ValueError('Unexpected frozen source path')
        data=original.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:
            data=subprocess.check_output(['git','show',plan['source_commit']+':'+name],cwd=ROOT)
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Neither working nor committed frozen source matches: '+name)
        restored=target/name;restored.parent.mkdir(parents=True,exist_ok=True);restored.write_bytes(data)
        inventory.append(dict(path=name,sha256=expected,bytes=len(data)))
    (directory/'source_snapshot.json').write_text(json.dumps(dict(files=inventory,
        all_bytes_verified=True,restore='Source snapshots preserve exact original bytes; replay from that frozen code revision.'),indent=2)+'\n')
    print(json.dumps(dict(directory=directory.name,files=len(inventory),all_bytes_verified=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directories',nargs='+')
    for name in parser.parse_args().directories:snapshot(name)
