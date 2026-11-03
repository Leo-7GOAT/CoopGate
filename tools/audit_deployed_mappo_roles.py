"""Completed development only:verify learned role participation in execution.

This does not establish a causal cooperative benefit or critic-over-IPPO
superiority. No new SUMO,policy update,independent initial,or risk inference.
"""
import json
from pathlib import Path
import numpy as np
from coopgate.episode_store_v21 import observations
import mappo_coupled_formal_experiment as e


def run():
    pilot=e.BASE/'development/history_power_pilot_20261103'
    owner=e.read(pilot/'owner_native_exit.json')
    if owner['actual_native_exit_code']!=0 or owner['report_sha256']!=e.sha(pilot/'report.json'):
        raise ValueError('Completed pilot actual native owner required')
    report=e.read(pilot/'report.json')
    if report['actual_sumo_executions']!=1280 or not report['all_raw_blocks_sealed_and_exact_replayed']:
        raise ValueError('All1280 SUMO outcomes must already be independently replayed')
    out=e.BASE/'deployed_role_participation_20261103';out.mkdir(exist_ok=False)
    totals={};protocols={};source={};rows=0
    for seed in ('4703','4704','4705'):
        folder=e.BASE/f'development/joint_{seed}';audit=e.read(folder/'audit.json');protocol=e.read(folder/'protocol.json')
        if (audit['algorithm']!='mappo_ctde_20261101' or audit['steps']!=2097152 or not audit['final_budget_complete']
            or not audit['on_policy_proposals_verified'] or not audit['independent_gae_verified']
            or audit['policy_sha256']!=e.sha(folder/'policy.pt') or not all(v>0 for v in audit['role_parameter_delta_squared_l2'])):
            raise ValueError('Genuine all-role joint PPO update/weight audit required')
        protocols[seed]=protocol['ppo_config']
        totals[seed]=dict(initials=0,active_decision_boundaries=[0]*3,proposal_differs_from_baseline=[0]*3,
            executed_differs_from_baseline=[0]*3,initials_with_executed_role_change=[0]*3,
            proposals_by_role=[set(),set(),set()],learning_steps=0,multi_role_execution_steps=0,
            role_parameter_delta_squared_l2=audit['role_parameter_delta_squared_l2'])
        source[seed]=dict(protocol_sha256=e.sha(folder/'protocol.json'),audit_sha256=e.sha(folder/'audit.json'),policy_sha256=e.sha(folder/'policy.pt'))
    for shard in range(4):
        for block in e.read(pilot/f'shard{shard}/report.json')['blocks']:
            db=pilot/f'shard{shard}'/block['block']/'observations.sqlite'
            if e.sha(db)!=block['database_sha256']:raise ValueError('Completed pilot data changed')
            for _,record,trace in observations(db,block['metadata'],block['keys']):
                if record['method']=='baseline':continue
                rows+=1;seed=record['method'];r=totals[seed];r['initials']+=1
                state=np.array(record['row']['initial']);ever=np.zeros(3,bool)
                for item in trace:
                    active=state[:,0]<60.;base=np.array(item['baseline_command']);proposal=np.array(item['actor_proposal']);command=np.array(item['command'])
                    changed=active&(np.abs(command-base)>1e-9)
                    if item['actual_learning_used']!=bool(changed.any()):raise ValueError('Actual adoption identity differs')
                    r['learning_steps']+=int(changed.any());r['multi_role_execution_steps']+=int(changed.sum()>=2);ever|=changed
                    for role in range(3):
                        r['active_decision_boundaries'][role]+=int(active[role])
                        r['proposal_differs_from_baseline'][role]+=int(active[role] and abs(proposal[role]-base[role])>1e-9)
                        r['executed_differs_from_baseline'][role]+=int(changed[role])
                        if active[role]:r['proposals_by_role'][role].add(float(proposal[role]))
                    state=np.array(item['x'])
                for role in range(3):r['initials_with_executed_role_change'][role]+=int(ever[role])
    if rows!=960 or any(r['initials']!=320 for r in totals.values()):raise ValueError('Full three-model320-case scope required')
    for r in totals.values():r['proposals_by_role']=[sorted(x) for x in r['proposals_by_role']]
    result=dict(completed=True,results=totals,source=source,ppo_configs=protocols,
        same_declared_ppo_configuration_across_three_models=len({json.dumps(x,sort_keys=True) for x in protocols.values()})==1,
        inference='Nonzero audited all-role parameter changes and measured role commands prove training/execution participation;they do not prove causal cooperation or centralized-critic superiority.',
        pilot_source_report_sha256=e.sha(pilot/'report.json'),analysis_source_sha256=e.sha(__file__),
        independent_reused_initials=320,reused_model_initial_records=960,new_sumo_runs=0,new_independent_initials=0,new_error_allocated=False)
    e.write(out/'report.json',result)
    print(json.dumps(dict(results=totals,same_declared_ppo_configs=result['same_declared_ppo_configuration_across_three_models'])),flush=True)


if __name__=='__main__':run()
