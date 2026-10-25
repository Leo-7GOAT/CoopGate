"""Pair the complete trained DMPS development cohort with the frozen baseline.

This measures this trained adapter's task utility, not method-family dominance,
matched training budgets, a gate-only causal effect, or a new risk certificate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv, matched_plant_parameters
import run_joint_risk_v42 as baseline
import train_dmps_observable_long_run as parent

ROOT=parent.ROOT
PARENT=ROOT/'artifacts/literature_reproduction/dmps_observable_long_run'
SOURCES=('tools/pair_dmps_recovery_baseline.py','tests/test_pair_dmps_recovery_baseline.py',
         'docs/dmps_recovery_quality_protocol.md','tools/run_joint_risk_v42.py')


class BaselineEnv(MatchedShiftedMergeEnv):
    def reset_disturbed(self,*args,**kwargs):
        result=super().reset_disturbed(*args,**kwargs)
        self.method='bounded_actuator_recovery_baseline'
        return result


def paired_outcome(baseline_row,learning_review,learning_finish):
    base_failed=bool(not baseline_row['completed'] or baseline_row['aborted'] or
        baseline_row['collision'] or baseline_row['contract_violations'])
    learned_failed=learning_review is None or not learning_review['completed']
    assessable=not base_failed and not learned_failed
    harm=None if base_failed else bool(learned_failed or
        any(a-b>1e-9 for a,b in zip(learning_review['per_vehicle_cost'],baseline_row['per_vehicle_cost'])) or
        any(a-b>1e-9 for a,b in zip(learning_finish,baseline_row['per_vehicle_finish'])))
    gain=100*(baseline_row['team_cost']-np.mean(learning_review['per_vehicle_cost']))/baseline_row['team_cost'] if assessable else None
    return dict(baseline_failed=base_failed,learned_failed=bool(learned_failed),
        relative_harm=harm,relative_performance_assessable=assessable,
        paired_cost_gain_percent=None if gain is None else float(gain))


def prerequisites():
    io=parent.base.io
    for name in ('complete.json','verification.json','observed_execution_exit.json'):io.committed(PARENT/name)
    done=io.read(PARENT/'complete.json');proof=io.read(PARENT/'verification.json')
    if (proof['complete_sha256']!=io.digest(PARENT/'complete.json') or proof['observed_exit']['exit_code']!=0
        or not proof['all_physics_and_replay_arrays_reconstructed'] or proof['training_actual_decisions']!=parent.STEPS
        or len(proof['evaluation'])!=10 or not proof['source_and_file_hashes_verified']):
        raise ValueError('Complete audited longer training and every evaluation condition required')
    for name,expected in done['files_sha256'].items():
        if io.digest(PARENT/name)!=expected:raise ValueError('Parent data changed: '+name)
    protocol=io.read(PARENT/'protocol.json')
    for name,expected in protocol['source_sha256'].items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Parent source changed: '+name)
    return protocol,proof


def run(output):
    io=parent.base.io
    protocol,proof=prerequisites()
    for name in SOURCES:io.committed(ROOT/name)
    hashes={**protocol['source_sha256'],**{name:io.digest(ROOT/name) for name in SOURCES}}
    identity=io.runtime_identity();torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=False)
    io.exclusive_json(output/'plan.json',dict(cases=parent.evaluation_cases(),source_sha256=hashes,
        parent_complete_sha256=io.digest(PARENT/'complete.json'),parent_verification_sha256=io.digest(PARENT/'verification.json'),
        runtime=identity,baseline='Unchanged MemoizedRecoveryGate(None, public plant class, baseline)',
        learned='Fixed fully trained original DMPS nominal-model adapter',
        comparative_family_superiority_claimed=False,new_risk_certificate=False,
        independent_initials=2,conditions=10,baseline_sumo_episodes=10))
    entries=[]
    for expected,learned in zip(parent.evaluation_cases(),proof['evaluation']):
        assert learned['case']==expected
        folder=output/f"{expected['index']:03d}";folder.mkdir()
        case=dict(expected,cluster_id=f"dmps_quality_{expected['index']:03d}",scenario='nominal')
        source_folder=PARENT/'evaluation'/f"episode_{expected['index']:03d}"
        source_episode=io.read(source_folder/'episode.json') if (source_folder/'episode.json').exists() else None
        failure=None;record=review=None
        with BaselineEnv(plant=case['plant']) as env:
            try:
                record=baseline.execute_episode(case,'baseline',None,env)
            except (RuntimeError,ValueError) as error:
                failure=dict(type=type(error).__name__,message=str(error),stage='baseline_execution')
                io.exclusive_json(folder/'failure.json',dict(case=case,failure=failure))
            if record is not None:
                # Actual data are persisted before any replay or comparison.
                io.exclusive_json(folder/'episode.json',record)
                if source_episode is not None:
                    np.testing.assert_allclose(record['row']['initial'],source_episode['initial'],atol=1e-8,rtol=0)
                case.update(initial_sha256=baseline.design.shared.array_sha256(np.asarray(record['row']['initial']),decimals=8),
                    dynamics_sha256=baseline.design.shared.canonical_sha256(matched_plant_parameters(case['seed'],case['plant'])),
                    disturbance_sha256=baseline.design.shared.array_sha256(disturbance_tape(case['seed'],case['disturbance'],.1,250)))
                review=baseline.audit_record(case,record,None)
                io.exclusive_json(folder/'review.json',review)
        outcome=paired_outcome(record['row'],learned['review'],source_episode['capped_finish'] if source_episode else None) if record else dict(
            baseline_failed=True,learned_failed=learned['review'] is None or not learned['review']['completed'],
            relative_harm=None,relative_performance_assessable=False,paired_cost_gain_percent=None)
        result=dict(case=expected,baseline_failure=failure,baseline_row=record['row'] if record else None,
            baseline_review=review,learning_review=learned['review'],**outcome)
        io.exclusive_json(folder/'complete.json',dict(result=result,
            files_sha256={p.name:io.digest(p) for p in folder.iterdir() if p.is_file()}))
        entries.append(dict(case=expected,complete_sha256=io.digest(folder/'complete.json'),result=result))
        print(json.dumps(dict(baseline_conditions=len(entries),expected_conditions=10)),flush=True)
    for name,expected in hashes.items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Bound source changed: '+name)
    assert io.runtime_identity()==identity
    io.exclusive_json(output/'complete.json',dict(cases=entries,plan_sha256=io.digest(output/'plan.json'),
        all_conditions_retained=True,new_risk_certificate=False,comparative_family_superiority_claimed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
