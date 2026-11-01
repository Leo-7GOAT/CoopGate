"""Complete fixed-model matrix; preserve repeated-case independence scope."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101/development'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize():
    baselines={};models={};inputs={};all_initials=set();commands=substeps=0
    for seed in (4703,4704,4705):
        directory=BASE/f'joint_{seed}_recovery_integration'
        plan=json.loads((directory/'plan.json').read_text())
        receipt=json.loads((directory/'native_receipt.json').read_text())
        proof=json.loads((directory/'verification.json').read_text())
        report=json.loads((directory/'report.json').read_text())
        complete=json.loads((directory/'collection_complete.json').read_text())
        if (receipt['collection_actual_native_exit_code']!=0 or receipt['review_actual_native_exit_code']!=0
                or not proof['all_match'] or proof['actual_sumo_episodes']!=400
                or complete['episodes_sha256']!=sha(directory/'episodes.jsonl')
                or proof['report_sha256']!=sha(directory/'report.json')):
            raise ValueError('Complete unchanged native/replay evidence required for every model')
        inputs[str(seed)]={n:sha(directory/n) for n in ('plan.json','episodes.jsonl','report.json','verification.json','native_receipt.json')}
        rows=[json.loads(s) for s in (directory/'episodes.jsonl').read_text().splitlines()]
        if len(rows)!=400:raise ValueError('Incomplete matrix')
        score_diagnostics={}
        for row in rows:
            key=tuple(row['key'][:4]);all_initials.add(tuple(row['key'][:2]))
            if row['row']['rule']=='baseline':
                physical={k:row['row'][k] for k in ('initial','plant','per_vehicle_finish',
                    'per_vehicle_variation','per_vehicle_cost','completed','aborted','collision','contract_violations')}
                physical['trace']=[{k:v for k,v in t.items() if k!='gate_ms'} for t in row['trace']]
                if key in baselines and baselines[key]!=physical:raise ValueError('Physical baseline changed across models')
                baselines[key]=physical
        for rule in ('nominal_performance','bank_performance'):
            values=np.array([t['finite_bank_performance_margin'] for r in rows if r['row']['rule']==rule
                             for t in r['trace'] if t['actual_learning_used']])
            score_diagnostics[rule]=dict(adopted_decisions=len(values),positive_margins=int((values>1e-10).sum()),
                zero_margins=int((np.abs(values)<=1e-10).sum()),
                warning='A minimum over unaffected/equal-cost coordinates can be identically zero; this is not calibrated confidence')
        models[str(seed)]=dict(policy_sha256=plan['model']['policy_sha256'],methods=report['methods'],
            score_diagnostics=score_diagnostics,original_bank_utility_threshold_met=report['methods']['bank_performance']['all_scope_cost_gain_percent']>.5)
        commands+=proof['commands_matched'];substeps+=proof['independent_physical_substeps']
    result=dict(fixed_models=[4703,4704,4705],actual_sumo_executions=1200,
        independent_initials=len(all_initials),repeated_conditions=80,baseline_physical_parity_all_models=True,
        commands_matched=commands,physical_substeps_matched=substeps,models=models,inputs_sha256=inputs,
        risk_certified=False,all_models_bank_utility_threshold_met=all(m['original_bank_utility_threshold_met'] for m in models.values()),
        inference='Same FIVE initials across all models; neither1200 nor15 independent trials. Model selection and risk certification are not justified.')
    path=ROOT/'artifacts/mappo_ctde_20261101/three_model_gate_matrix.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','inputs_sha256')}),flush=True)


if __name__=='__main__':summarize()
