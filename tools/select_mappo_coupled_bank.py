"""Deterministic development set cover over complete first-prefix witnesses."""
import json
import zlib
import mappo_formal_experiment as e
from probe_mappo_coupled_witnesses import OUT,PROFILES

DEST=e.BASE/'development/compact_coupled_bank_20261102.json'


def run():
    owner=e.read(OUT/'owner_native_exit.json');report=e.read(OUT/'report.json')
    if owner['actual_native_exit_code']!=0 or owner['report_sha256']!=e.sha(OUT/'report.json') or not report['completed']:
        raise ValueError('Complete owned probe required')
    result={}
    for profile in PROFILES:
        path=OUT/profile/'all_paths.json.zlib'
        if e.sha(path)!=report['profiles'][profile]['all_paths_sha256']:raise ValueError('Probe paths changed')
        data=json.loads(zlib.decompress(path.read_bytes()));incidents=data['incidents']
        remaining=set(range(len(incidents)));chosen=[]
        while remaining:
            count,_,index=max((sum(k in incidents[i]['prefixes'][0]['witness_paths'] for i in remaining),-k,k)
                              for k in range(len(data['library'])))
            if not count:break
            covered=sorted(i for i in remaining if index in incidents[i]['prefixes'][0]['witness_paths'])
            chosen.append(dict(path_id=index,**data['library'][index],newly_covered_incidents=covered))
            remaining.difference_update(covered)
        result[profile]=dict(paths=chosen,development_incidents=len(incidents),
            covered=len(incidents)-len(remaining),uncovered=[dict(cluster_id=incidents[i]['cluster_id'],method=incidents[i]['method']) for i in sorted(remaining)])
    e.write(DEST,dict(selection='greedy complete cost/phase first-prefix witness coverage; tie lowest library index',
        probe_report_sha256=e.sha(OUT/'report.json'),selector_sha256=e.sha(__file__),profiles=result,
        development_data_only=True,alpha_allocated=False,closed_loop_repair_demonstrated=False))
    print(json.dumps({p:dict(path_ids=[x['path_id'] for x in r['paths']],covered=r['covered'],total=r['development_incidents']) for p,r in result.items()}))


if __name__=='__main__':run()
