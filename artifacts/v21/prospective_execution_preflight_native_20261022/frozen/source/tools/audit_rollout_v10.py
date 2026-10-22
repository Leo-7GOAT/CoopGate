"""Post-freeze differential audit, not additional independent performance data."""
import json
from pathlib import Path
import time
import numpy as np
from coopgate.rollout_v10 import RolloutGate
from coopgate.learning_v6 import load_models,predict
from coopgate.merge_v4 import MergeConfig

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v10'


def main():
    protocol=json.loads((D/'protocol.json').read_text());config=MergeConfig(**protocol['config'])
    initials=json.loads((D/'study/initial_states.json').read_text());models=load_models(ROOT/'artifacts/v6',protocol['model_seeds'])
    records=[];command_mismatches=0;value_mismatches=0
    for scenario,seeds in protocol['test_seeds'].items():
        initial=np.array(next(r['state'] for r in initials if r['seed']==seeds[0]))
        proposal=int(predict(models,initial[None])[0][0])
        for mode in protocol['disturbances']:
            trace=json.loads((D/f'study/trace_{scenario}_{mode}_rollout_learned.json').read_text())
            gates=[RolloutGate(initial,np.zeros(3),config,protocol['radius'],'pair',proposal,pruning=p) for p in (True,False)]
            # Warm both variants before timing. This audit excludes model inference and initial selection.
            for gate in gates:gate.select(initial,np.zeros(3),None)
            for k in range(0,len(trace),2):
                x=initial if k==0 else np.array(trace[k-1]['x'])
                prev=np.zeros(3) if k==0 else np.array(trace[k-1]['command'])
                outputs=[];elapsed=[[],[]]
                for repetition in range(5):
                    for index in ((0,1) if repetition%2==0 else (1,0)):
                        gate=gates[index];gate.step=k
                        start=time.perf_counter();command,info=gate.select(x,prev,None)
                        elapsed[index].append((time.perf_counter()-start)*1000)
                        if repetition==0:outputs.append((index,command,info))
                outputs.sort(key=lambda t:t[0]);fast,slow=outputs
                command_mismatches+=not np.array_equal(fast[1],slow[1])
                value_mismatches+=any(fast[2][key]!=slow[2][key] for key in ('base_ticks','selected_ticks','active_profile'))
                records.append({'scenario':scenario,'disturbance':mode,'step':k,
                    'pruned_tail_steps':fast[2]['tail_feedback_steps'],'unpruned_tail_steps':slow[2]['tail_feedback_steps'],
                    'pruned_median_ms':float(np.median(elapsed[0])),'unpruned_median_ms':float(np.median(elapsed[1]))})
    result={'scope':'Every second decision state from the eight predesignated first-seed V10 target traces; correlated diagnostic states, not independent test scenes.',
        'states':len(records),'command_mismatches':int(command_mismatches),'value_mismatches':int(value_mismatches),
        'pruned_tail_steps':sum(r['pruned_tail_steps'] for r in records),'unpruned_tail_steps':sum(r['unpruned_tail_steps'] for r in records),
        'pruned_state_median_p95_ms':float(np.percentile([r['pruned_median_ms'] for r in records],95)),
        'unpruned_state_median_p95_ms':float(np.percentile([r['unpruned_median_ms'] for r in records],95)),
        'records':records}
    (D/'pruning_audit.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
