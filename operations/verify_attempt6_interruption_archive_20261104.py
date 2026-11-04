"""Independent storage verification; does not certify partial formal outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'inventory.json').read_bytes())
    owner = json.loads((directory/'owner_native_exit.json').read_bytes())
    if owner['actual_native_exit_code'] != 0 or owner['inventory_sha256'] != sha(directory/'inventory.json'):
        raise ValueError('Actual independent archive-completion witness required')
    if (not manifest['completed'] or manifest['recovered_formal_study_complete']
            or manifest['original_native_exit_code'] is not None
            or manifest['retained_actual_records'] != 642
            or manifest['new_required_planned_runs'] != 58750
            or manifest['new_error_allocated']):
        raise ValueError('Archive scope or interrupted-owner disclosure changed')
    rows=manifest['inventory'];paths=[q['path'] for q in rows]
    if len(paths)!=len(set(paths)) or len(rows)!=manifest['files']:
        raise ValueError('Archive inventory membership differs')
    seen=set()
    for part in manifest['parts']:
        path=directory/part['name']
        if path.parent.resolve()!=directory.resolve() or path.stat().st_size!=part['bytes'] or sha(path)!=part['sha256']:
            raise ValueError('Archive part identity or bytes differ')
        members={q['path']:q for q in rows if q['part']==part['name']}
        with zipfile.ZipFile(path) as z:
            if len(z.namelist())!=len(members) or set(z.namelist())!=set(members):
                raise ValueError('Part has missing, duplicate or foreign members')
            for name,q in members.items():
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe extraction path')
                data=z.read(name)
                if len(data)!=q['bytes'] or hashlib.sha256(data).hexdigest()!=q['sha256']:
                    raise ValueError('Restored archive member bytes differ')
                seen.add(name)
    if seen!=set(paths) or sum(q['bytes'] for q in rows)!=manifest['original_bytes']:
        raise ValueError('Whole archive accounting differs')
    return dict(storage_archive_verified=True,files=len(rows),parts=len(manifest['parts']),
        original_bytes=manifest['original_bytes'],original_native_exit_code=None,
        completed_formal_results_certified=False,primary_effects_summarized=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.directory)))
