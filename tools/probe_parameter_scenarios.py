"""Off-policy recorded-prefix diagnostic; no new SUMO efficacy claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from coopgate.mappo_deployment import load_deployment_actor
from coopgate.parameter_scenario_gate import ParameterScenarioGate

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101/development'


def run():
    torch.set_num_threads(1);rows={}
    for i in range(4):
        p=BASE/f'scenario_score_grid_20261102_shard{i}'
        if not json.loads((p/'verification.json').read_text())['all_match']:raise ValueError('Reviewed source required')
        for line in (p/'episodes.jsonl').read_text().splitlines():
            row=json.loads(line);rows[tuple(row['key'])]=row
    model,identity=load_deployment_actor(BASE/'joint_4705/policy.pt')
    oracle=json.loads((ROOT/'artifacts/mappo_ctde_20261101/scenario_forecast_error_diagnosis_20261102.json').read_text())
    labels={(tuple(e['case']),e['step']):e['oracle_detects_regression'] for e in oracle['examples']}
    examples=[]
    for key,record in rows.items():
        if key[-1]!='confidence_4705_0' or key[0]!=96902001 or key[2]!='lag_strong' or key[3] not in ('clean','uniform','pulse'):continue
        gate=ParameterScenarioGate(model,'lag_strong');state=np.array(record['row']['initial']);prev=np.zeros(3)
        for step,item in enumerate(record['trace']):
            command,info=gate.select(state,prev)
            if item['actual_learning_used']:
                examples.append(dict(case=list(key[:4]),step=step,
                    old_command=item['command'],new_prefix_probe_command=command.tolist(),
                    unchanged_command=bool(np.array_equal(command,item['command'])),
                    old_oracle_regression=labels.get((key[:4],step)),
                    new_learning_used=info['actual_learning_used'],score=info['scenario_pass_fraction'],
                    parameter_path_rejections=sum(not e['passed'] and e['checked_paths']>33 for e in info['candidate_evidence'])))
            # Feed the actually executed historical command, NOT the new probe output.
            prev=np.array(item['command']);state=np.array(item['x'])
    if len(examples)!=4:raise ValueError('Full preregistered four-decision prefix scope required')
    result=dict(scope='off-policy recorded-prefix diagnostic',closed_loop_sumo_runs=0,extra_independent_initials=0,
        policy=identity,online_privileged_inputs=False,oracle_labels_used_only_for_reporting=True,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ('src/coopgate/parameter_scenario_gate.py','tools/probe_parameter_scenarios.py','docs/parameter_scenario_20261102_protocol.md')},examples=examples)
    out=ROOT/'artifacts/mappo_ctde_20261101/parameter_scenario_prefix_probe_20261102.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(decisions=len(examples),changed_adverse=sum(e['old_oracle_regression'] is True and not e['unchanged_command'] for e in examples),
                         changed_nonadverse=sum(e['old_oracle_regression'] is False and not e['unchanged_command'] for e in examples))),flush=True)


if __name__=='__main__':run()
