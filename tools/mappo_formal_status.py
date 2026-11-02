"""Read progress receipts only;never inspect partial scientific outcomes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'artifacts/mappo_ctde_20261101/formal_mappo_attempt3_20261102'


def status():
    execution=ROOT/'execution';result={};collected=reviewed=0
    for stage in ('calibration','confirmation'):
        for action,key in (('collect','completed'),('review','reviewed')):
            counts=[]
            for shard in range(4):
                path=execution/f'{stage}_{action}_{shard}.log';latest=0
                if path.exists():
                    for line in path.read_text(encoding='utf-8').splitlines():
                        try:item=json.loads(line)
                        except json.JSONDecodeError:continue
                        latest=max(latest,int(item.get(key,0)))
                receipt=ROOT/stage/f'shard{shard}'/('collection_complete.json' if action=='collect' else 'verification.json')
                if receipt.exists():latest=int(json.loads(receipt.read_bytes())['executions'])
                counts.append(latest)
            result[stage+'_'+action]=counts
            if action=='collect':collected+=sum(counts)
            else:reviewed+=sum(counts)
    events_path=execution/'events.jsonl';events=[]
    if events_path.exists():
        for line in events_path.read_text(encoding='utf-8').splitlines():
            try:item=json.loads(line)
            except json.JSONDecodeError:continue
            if 'actual_native_exit_code' in item:events.append({k:v for k,v in item.items() if k in ('stage','action','shard','native_pid','actual_native_exit_code')})
    result.update(actual_collected=collected,full_independently_reviewed=reviewed,total_scheduled=37888,
        collection_fraction=collected/37888,native_terminal_events=events,
        full_completion_receipt_exists=(ROOT/'completion_receipt.json').exists(),partial_scientific_outcomes_read=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':status()
