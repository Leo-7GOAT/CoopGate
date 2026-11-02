"""Complete paired optimization seeds,including unfavorable local-critic results."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101/development'


def summarize():
    members={}
    for seed in (4703,4704,4705):
        execution=ROOT/f'artifacts/mappo_ctde_20261101/ippo_{seed}_execution'
        complete=json.loads((execution/'completion.json').read_text())
        if any(e['actual_native_exit_code']!=0 for e in complete['native_commands']):raise ValueError('Every native stage required')
        own=BASE/f'ippo_{seed}';audit=json.loads((own/'audit.json').read_text())
        digest=hashlib.sha256((own/'policy.pt').read_bytes()).hexdigest()
        if digest!=complete['policy_sha256'] or digest!=audit['policy_sha256']:raise ValueError('Policy changed')
        proof=json.loads((BASE/f'ippo_{seed}_sumo_integration/verification.json').read_text())
        report=json.loads((BASE/f'ippo_{seed}_exact/report.json').read_text())
        if not proof['all_match'] or proof['actual_sumo_episodes']!=88 or report['protocol']['expected_executions']!=5632:
            raise ValueError('Complete fixed comparison/replay required')
        members[str(seed)]=dict(mappo_vs_local=report['results'][f'joint_central_{seed}']['comparisons'][f'joint_local_{seed}'],
            ippo_vs_strong_team=report['results'][f'joint_local_{seed}']['comparisons']['team_planner'],
            initialization=json.loads((own/'paired_initialization_audit.json').read_text()),
            policy_sha256=digest,replay=proof)
    result=dict(fixed_seeds=[4703,4704,4705],members=members,
        all_mappo_gain_lower_bounds_positive=all(m['mappo_vs_local']['team_gain_one_sided_t_lower_s']>0 for m in members.values()),
        inference='Paired development optimization seeds conditional on fixed I parent. Not from-scratch sample efficiency or a claim over arbitrary training seeds. Critics differ in parameter count/compute.',
        additional_independent_initials_from_replay=0)
    (ROOT/'artifacts/mappo_ctde_20261101/ippo_matched_family_20261102.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({s:m['mappo_vs_local'] for s,m in members.items()},indent=2),flush=True)


if __name__=='__main__':summarize()
