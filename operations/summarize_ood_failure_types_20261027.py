"""Describe all verified OOD failure flags without adding inferential tests."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

from summarize_completed_ood_20261027 import read_complete, SOURCE, ROOT, digest

OUT=ROOT/'artifacts/recovery_20261027/ood_failure_types'


def bucket():
    return dict(counts=Counter(),failure_reasons=Counter(),abort_reasons=Counter())


def main():
    report=read_complete()
    if OUT.exists():raise FileExistsError('Preserve existing complete diagnostic output')
    execution=SOURCE/'execution'
    receipt=json.loads((execution/'exit.json').read_text(encoding='utf-8'))
    path=execution/'blocks_manifest.json'
    if digest(path)!=receipt['blocks_manifest_sha256']:
        raise ValueError('Complete collection manifest changed')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    methods=defaultdict(bucket);cells=defaultdict(Counter)
    for index,entry in enumerate(manifest['entries']):
        if entry['block_index']!=index:raise ValueError('Frozen block order changed')
        path=SOURCE/'blocks'/f'{index:05d}'/'observations.sqlite'
        if digest(path)!=entry['database_sha256']:raise ValueError('Sealed data changed')
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            rows=list(db.execute('SELECT position,case_key,row_blob,row_sha256 FROM episodes ORDER BY position'))
        if len(rows)!=13 or [r[0] for r in rows]!=list(range(13)) or len({r[1] for r in rows})!=13:
            raise ValueError('Complete unique original method family required')
        for _,key,blob,expected in rows:
            raw=zlib.decompress(blob)
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Stored row changed')
            row=json.loads(raw)['row'];key=json.loads(key)
            if key[-2:]!=[row['rule'],row['model']]:raise ValueError('Method identity differs')
            item=methods[(row['rule'],row['model'])]
            counts=item['counts'];counts['episodes']+=1
            for name in ('failure','aborted','collision'):
                counts[name]+=bool(row[name])
            counts['not_completed']+=not row['completed']
            counts['episodes_with_route_violation']+=row['contract_violations']>0
            counts['route_violation_substeps']+=row['contract_violations']
            item['failure_reasons'].update(row['failure_reasons'])
            if row['aborted']:item['abort_reasons'][row['abort_reason']]+=1
            cell=(row['rule'],row['model'],row['scenario'],row['plant_name'],row['perception_name'])
            cells[cell]['episodes']+=1;cells[cell]['failures']+=bool(row['failure'])
    if len(methods)!=13 or sum(v['counts']['episodes'] for v in methods.values())!=22464:
        raise ValueError('Full 22464-run diagnostic coverage required')
    for c in report['cells']:
        key=tuple(c[n] for n in ('rule','model','regime','plant','perception'))
        if cells[key]['episodes']!=c['episodes'] or cells[key]['failures']!=c['failures']:
            raise ValueError('Failure counts differ from the independently replayed report')
    value=dict(source_report_sha256=digest(SOURCE/'analysis/report.json'),
        source_blocks_manifest_sha256=digest(execution/'blocks_manifest.json'),
        actual_sumo_runs=22464,independent_initials=192,new_tests=0,new_sumo_runs=0,
        failure_flags_are_nonexclusive=True,recorded_collision_is_not_the_composite_failure_rate=True,
        methods=[dict(rule=k[0],model=k[1],**{n:dict(v) for n,v in d.items()})
                 for k,d in sorted(methods.items())])
    OUT.mkdir(parents=True)
    (OUT/'summary.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    (OUT/'sha256.json').write_text(json.dumps({'summary.json':digest(OUT/'summary.json')})+'\n',encoding='utf-8')
    print(json.dumps([m for m in value['methods'] if m['rule'] in ('baseline','bank_performance')]),flush=True)


if __name__=='__main__':main()
