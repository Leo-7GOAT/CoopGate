"""Add at most four certified rank-two commands to the existing candidate set.

All three fixed actors use their own local observations. Raw masked logit
ordering supplies action preferences, never feasibility probabilities. The
central critic is not called. The original selector, strict phase, forecast
bank and gain floor remain; an optional inherited variation reserve remains.
"""
from types import FunctionType

import numpy as np
import torch
from coopgate.ego_v13 import ego_observations
from coopgate.marl_v12 import LEVELS,action_mask
from coopgate.recovery_joint_gate_v35 import compositions,project_command
from experiments.joint_admissibility import gate as integrated
from experiments.switching_reserve.gate import SwitchingReserveGate


def local_top_two(model,observations,mask):
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=(3,len(LEVELS)) or not mask.any(axis=1).all():
        raise ValueError('Three nonempty legal local action supports required')
    device=next(model.parameters()).device
    with torch.no_grad():
        obs=torch.as_tensor(observations,dtype=torch.float32,device=device)
        raw=model.actor(model.features(obs)).detach().cpu().numpy()
    if raw.shape!=mask.shape or not np.isfinite(raw).all():
        raise ValueError('Finite three-role local logits required')
    indices=np.empty((3,2),dtype=int);scores=np.empty((3,2))
    for role in range(3):
        legal=np.flatnonzero(mask[role])
        order=legal[np.argsort(-raw[role,legal],kind='stable')]
        selected=order[:2] if len(order)>1 else np.repeat(order,2)
        indices[role]=selected;scores[role]=raw[role,selected]
    return indices,scores


class TopTwoMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        original=integrated.JointAdmissibilityGate._select
        if original.__code__.co_freevars:raise ValueError('Unexpected selector closure')
        scope=dict(original.__globals__);scope['compositions']=self._candidate_union
        self._local_selector=FunctionType(original.__code__,scope,
            name='bounded_local_top_two_selector',argdefs=original.__defaults__)

    def select(self,state,previous,unused=None):
        self._top_two_evidence=None
        command,info=super().select(state,previous,unused)
        info.update(candidate_proposal_source='local_masked_top_two',
                    top_two_candidates=self._top_two_evidence,
                    maximum_extra_candidates=4,logits_are_feasibility_probabilities=False,
                    deployed_critic_used=False)
        return command,info

    def _select(self,state,previous,unused=None):
        # The copied selector has exactly one new composition provider. Its
        # function globals are private; frozen modules are never monkeypatched.
        baseline=self.baseline;parent=self
        class CaptureBaseline:
            def __getattr__(self,name):return getattr(baseline,name)
            def select(self,*args):
                command,info=baseline.select(*args)
                parent._current_public_baseline=np.asarray(command).copy()
                return command,info
        self.baseline=CaptureBaseline()
        try:return self._local_selector(self,state,previous,unused)
        finally:self.baseline=baseline

    def _candidate_union(self,state,previous,proposal,baseline,mlo,mhi,bounds,packed):
        candidates,original_count=compositions(state,previous,proposal,baseline,mlo,mhi,bounds,packed)
        mask=action_mask(previous,self.c)
        indices,scores=local_top_two(self.model,ego_observations(state,previous,self.step,self.c),mask)
        active=state[:,0]<self.c.goal
        if not np.array_equal(LEVELS[indices[:,0]][active],np.asarray(proposal)[active]):
            raise ValueError('Rank-one local proposals must equal the original greedy actions')
        seen=[np.asarray(baseline).copy()]+[row[2].copy() for row in candidates]
        extras=[];second=LEVELS[indices[:,1]].copy();second[~active]=baseline[~active]
        for role in range(4):
            if role<3 and not active[role]:continue
            target=np.asarray(baseline).copy()
            if role<3:target[role]=second[role]
            else:target=second.copy()
            repaired,vertex=project_command(state,previous,target,baseline,mlo,mhi,bounds,packed)
            if any(np.array_equal(repaired,prior) for prior in seen):continue
            seen.append(repaired.copy());code=100+role
            candidates.append((code,1.,repaired.copy(),vertex))
            extras.append(dict(composition=code,role=role if role<3 else 'all',
                requested=target.tolist(),command=repaired.tolist(),projection_vertex=int(vertex)))
        self._top_two_evidence=dict(indices=indices.tolist(),raw_logits=scores.tolist(),
            legal_counts=mask.sum(axis=1).tolist(),previous=np.asarray(previous).tolist(),
            active=active.tolist(),
            greedy_original_count=original_count,extra_candidates=extras,
            second_choice_is_local_action_preference=True)
        return candidates,original_count+len(extras)


class TopTwoHistoryGate(TopTwoMixin,integrated.JointAdmissibilityGate):
    pass


class TopTwoReserveGate(TopTwoMixin,SwitchingReserveGate):
    pass
