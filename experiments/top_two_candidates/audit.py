"""Check ranked proposal provenance, then replay the complete frozen physics."""
from types import FunctionType

import numpy as np
from coopgate.marl_v12 import LEVELS,action_mask
from coopgate.merge_v4 import MergeConfig
from experiments.switching_reserve import audit as reserved


def validate(item):
    if item.get('candidate_proposal_source')!='local_masked_top_two':
        return reserved.validate(item)
    if item['deployed_critic_used'] or item['logits_are_feasibility_probabilities']:
        raise ValueError('Local actor preferences must not claim a deployed critic or safety probabilities')
    evidence=item['top_two_candidates']
    indices=np.asarray(evidence['indices']);scores=np.asarray(evidence['raw_logits'])
    if indices.shape!=(3,2) or not np.issubdtype(indices.dtype,np.integer):
        raise ValueError('Exactly two ranked indices per role required')
    if scores.shape!=(3,2) or not np.isfinite(scores).all():raise ValueError('Finite local preference scores required')
    if np.any(indices<0) or np.any(indices>=len(LEVELS)):raise ValueError('Ranked action index outside support')
    mask=action_mask(np.asarray(evidence['previous'],dtype=float),MergeConfig())
    if not mask[np.arange(3)[:,None],indices].all():raise ValueError('Ranked action violates common mask')
    if evidence['legal_counts']!=mask.sum(axis=1).tolist():raise ValueError('Legal support size differs')
    if not np.array_equal(indices[:,0],item['agent_actions']):raise ValueError('Original greedy action ordering differs')
    if np.any(scores[:,0]<scores[:,1]):raise ValueError('Local preference ranking is reversed')
    ties=scores[:,0]==scores[:,1]
    if np.any(indices[ties,0]>indices[ties,1]):raise ValueError('Equal scores must use ascending index order')
    if np.any((indices[:,0]==indices[:,1])&(mask.sum(axis=1)>1)):
        raise ValueError('Repeated rank requires singleton legal support')
    extras=evidence['extra_candidates'];codes=[r['composition'] for r in extras]
    if len(extras)>4 or len(set(codes))!=len(codes) or any(c not in range(100,104) for c in codes):
        raise ValueError('At most four distinct ranked alternatives required')
    baseline=np.asarray(item['baseline_command']);active=np.asarray(evidence['active'],dtype=bool)
    if active.shape!=(3,):raise ValueError('Active-role identity required')
    second=LEVELS[indices[:,1]].copy();second[~active]=baseline[~active]
    commands=[]
    for extra in extras:
        role=extra['composition']-100;target=baseline.copy()
        if role<3:
            if not active[role]:raise ValueError('Cleared role must not manufacture an alternative')
            target[role]=second[role]
        else:target=second.copy()
        if not np.array_equal(target,extra['requested']):raise ValueError('Rank-two request provenance differs')
        command=np.asarray(extra['command'],dtype=float)
        if command.shape!=(3,) or not np.isfinite(command).all():raise ValueError('Finite extra command required')
        if not np.array_equal(command[~active],baseline[~active]):raise ValueError('Cleared vehicle command differs')
        if np.array_equal(command,baseline) or any(np.array_equal(command,c) for c in commands):
            raise ValueError('Extra commands must be distinct from baseline and each other')
        commands.append(command)
    if item['actual_learning_used'] and item['composition']>=100:
        selected=next((r for r in extras if r['composition']==item['composition']),None)
        if selected is None or not np.array_equal(selected['command'],item['command']):
            raise ValueError('Actual ranked command differs from its proposal provenance')
    reserved.validate(item)


_original=reserved.previous.original
_scope=dict(_original.__dict__);_scope['validate_prefix_evidence']=validate
_replay=FunctionType(_original.audit_record.__code__,_scope,
    name='independent_local_top_two_replay',argdefs=_original.audit_record.__defaults__)
_replay.__kwdefaults__=_original.audit_record.__kwdefaults__.copy()


def audit_record(case,record,model,*,reference_factory):
    return _replay(case,record,model,reference_factory=reference_factory)
