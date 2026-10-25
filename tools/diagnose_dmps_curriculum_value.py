"""Post-exit, descriptive value diagnosis of the failed curriculum experiment.

No refitting, selection, additional simulator sample or efficacy test occurs.
Equivalent commands are checked against the immutable nominal transition model.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.dmps_nominal_model import command_for
from coopgate.dmps_nominal_model_v2 import NominalModelEnvV2
from coopgate.merge_v4 import MergeConfig
import load_original_dmps as author
import verify_dmps_curriculum as audit

ROOT=audit.run.ROOT
PARENT=audit.OUT
OUT=PARENT.parent/'dmps_curriculum_value_diagnosis.json'
SOURCE=ROOT.parent/'reference_sources/dmps-f37df2d'


def discounted_returns(rewards,gamma=.99):
    result=[];value=0.
    for reward in reversed(rewards):
        value=float(reward)+gamma*value;result.append(value)
    return result[::-1]


def equivalent_actions(action,state,previous,config):
    """All one-coordinate {-1,0,1} probes that decode to the same command."""
    action=np.asarray(action,dtype=float)
    command=command_for(action,state,previous,config)
    result=[]
    for dim,value in itertools.product(range(3),(-1.,0.,1.)):
        alternative=action.copy();alternative[dim]=value
        if np.array_equal(command_for(alternative,state,previous,config),command):
            result.append((dim,alternative))
    return result


def replay_equivalent(model,state,previous,index,action,alternatives):
    model.sync(state,previous,index);snapshot=model.get_state()
    reference=model.step(action)
    for _,alternative in alternatives:
        model.set_state(snapshot);current=model.step(alternative)
        np.testing.assert_array_equal(current[0],reference[0])
        assert current[1:]==reference[1:]
    return len(alternatives)


def diagnose():
    io=audit.io;done=io.read(PARENT/'complete.json');proof=io.read(PARENT/'verification.json')
    assert proof['complete_sha256']==io.digest(PARENT/'complete.json')
    assert proof['all_physics_and_replay_arrays_reconstructed']
    assert proof['observed_exit']['exit_code']==0
    for name,expected in done['files_sha256'].items():assert io.digest(PARENT/name)==expected
    module,identity=author.load(SOURCE);config=MergeConfig()
    phases={str(i):dict(episodes=0,completed=0,actual_decisions=0) for i in range(4)}
    training=io.read(PARENT/'training_episodes.json')['episodes']
    for i,episode in enumerate(training):
        phase=io.read(PARENT/'training'/f'episode_{i:03d}'/'initialization.json')['phase']
        row=phases[str(phase)];row['episodes']+=1
        row['completed']+=int(episode['review']['completed'])
        row['actual_decisions']+=episode['review']['actual_decisions']
    rows=[];aggregate={}
    for role,checkpoint in [('new',PARENT/'trained_state.pt'),('old',audit.run.OLD/'trained_state.pt')]:
        model=module.Model(10,3,torch.device('cpu'),ret_low=-32.,ret_high=0.)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
        nominal=NominalModelEnvV2(np.array([[-140.,12.],[-220.,10.],[-190.,12.]]))
        errors=[];spans1=[];spans2=[];probes=states=0;exact_nominal_cases=0
        for case_index in range(10):
            episode=io.read(PARENT/f'evaluation_{role}'/f'episode_{case_index:03d}'/'episode.json')
            calls=[io.read(PARENT/f'evaluation_{role}'/f'call_{case_index*50+j:04d}.json')
                for j in range(len(episode['decisions']))]
            assert len(calls)==50 and calls[-1]['done']
            returns=discounted_returns([c['reward'] for c in calls])
            deterministic=episode['case']['plant']=='nominal' and episode['case']['disturbance']=='clean'
            if deterministic:exact_nominal_cases+=1
            for step,(decision,call,realized_return) in enumerate(zip(episode['decisions'],calls,returns)):
                x=np.asarray(decision['observation_state']);previous=np.asarray(decision['previous_command'])
                action=np.asarray(call['executed']);obs=torch.as_tensor(call['observation'],dtype=torch.float32)
                assert call['intervention']==0 # Continuation is exactly the recorded actor in this dataset.
                alternatives=equivalent_actions(action,x,previous,config)
                probes+=replay_equivalent(nominal,x,previous,step,action,alternatives)
                alternatives=[(None,action)]+alternatives
                actions=torch.as_tensor(np.stack([a for _,a in alternatives]),dtype=torch.float32)
                observations=obs.repeat(len(alternatives),1)
                with torch.no_grad():
                    q1=model.critic1(observations,actions).squeeze(-1).numpy()
                    q2=model.critic2(observations,actions).squeeze(-1).numpy()
                span1=float(np.ptp(q1));span2=float(np.ptp(q2))
                if len(alternatives)>1:
                    spans1.append(span1);spans2.append(span2);states+=1
                error=float(q1[0])-realized_return
                if deterministic:errors.append(error)
                rows.append(dict(role=role,case_index=case_index,step=step,
                    same_command_alternatives=len(alternatives)-1,q1=float(q1[0]),q2=float(q2[0]),
                    q1_equivalent_span=span1,q2_equivalent_span=span2,
                    realized_discounted_return=realized_return,
                    deterministic_nominal_continuation=deterministic,
                    nominal_q1_error=error if deterministic else None))
        aggregate[role]=dict(evaluation_states=500,states_with_equivalent_probes=states,
            nominal_transition_probes=probes,q1_equivalent_span_mean=float(np.mean(spans1)),
            q1_equivalent_span_max=max(spans1),q2_equivalent_span_mean=float(np.mean(spans2)),
            q2_equivalent_span_max=max(spans2),deterministic_nominal_episodes=exact_nominal_cases,
            nominal_q1_error_mean=float(np.mean(errors)),nominal_q1_error_max_abs=max(map(abs,errors)))
    return dict(parent_complete_sha256=io.digest(PARENT/'complete.json'),
        parent_audit_sha256=io.digest(PARENT/'verification.json'),
        source_sha256=io.digest(Path(__file__)),author_sources=identity,training_phases=phases,
        aggregate=aggregate,rows=rows,posthoc_descriptive_diagnosis=True,
        causal_training_failure_proven=False,new_simulator_executions=0,
        new_risk_certificate=False,comparative_family_superiority_claimed=False)


if __name__=='__main__':
    torch.set_num_threads(1)
    result=diagnose();audit.io.exclusive_json(OUT,result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
