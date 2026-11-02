"""Verify integrated extra-path forecasts exactly match the full prefix probe."""
import json
import zlib
import numpy as np
from coopgate.compact_coupled_gate import CompactCoupledGate
from coopgate.coupled_corner_witness import history_memory
from select_mappo_coupled_bank import DEST
import probe_mappo_coupled_witnesses as p


def run():
    checked=covered=0;issues=p.e.read(p.DIAG)['all_target_harmful_adoptions']
    for profile in p.PROFILES:
        data=json.loads(zlib.decompress((p.OUT/profile/'all_paths.json.zlib').read_bytes()))
        selected=[x for x in issues if x['plant']==profile]
        for issue,incident in zip(selected,data['incidents']):
            record=p.load_block(issue)[(issue['cluster_id'],issue['method'])]
            a=issue['admissions'][0];step=a['step'];entry=record['trace'][step]
            # Predict consumes no actor; a sentinel satisfies constructor's policy guard.
            gate=CompactCoupledGate(object(),profile,DEST);gate.step=step
            assert len(gate.tapes)==(33 if profile=='nominal' else 41)+len(gate.coupled_ids)
            for i,(_,_,rho) in enumerate(gate.parameter_scenarios):
                gate.parameter_memories[i]=history_memory([x['command'] for x in record['trace'][:step]],rho)
            for i,k in enumerate(gate.coupled_ids):
                path=len(gate.tapes)-len(gate.coupled_ids)+i
                args=(np.array(a['state_before']),np.array(a['previous']),
                      np.array(entry['actuator_memory_lower']),np.array(entry['actuator_memory_upper']))
                before=gate.predict(*args,None,path);after=gate.predict(*args,np.array(a['command']),path)
                old=incident['prefixes'][0]['paths'][k]
                assert bool(before[2])==old['baseline_complete'] and bool(after[2])==old['candidate_complete']
                if old['differences'] is not None:assert np.array_equal(after[0]-before[0],old['differences'])
                checked+=1
            covered+=bool(set(gate.coupled_ids)&set(incident['prefixes'][0]['witness_paths']))
    if covered!=28:raise ValueError('Integrated library lost a complete historical witness')
    report=dict(exact_integrated_prediction_pairs=checked,first_prefix_witnesses_preserved=covered,
        old_paths_retained=True,controller_sha256=p.e.sha(p.e.ROOT/'src/coopgate/compact_coupled_gate.py'),
        bank_sha256=p.e.sha(DEST),checker_sha256=p.e.sha(__file__),new_sumo_runs=0,deployment_certified=False)
    p.e.write(p.e.BASE/'development/compact_coupled_integration_20261102.json',report)
    print(json.dumps(report))


if __name__=='__main__':run()
