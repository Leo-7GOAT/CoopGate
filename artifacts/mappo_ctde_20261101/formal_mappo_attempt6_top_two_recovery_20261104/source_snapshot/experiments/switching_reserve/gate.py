"""Charge a parameter-free two-command variation reserve before learning use.

lambda*(|u-p|+|b-u|-|b-p|) is the exact additional command-variation cost
of u followed by the current backup b, relative to going directly from p to b.
It equals 2*lambda*distance(u,[p,b]). This LOCAL identity is not a bound on
unknown full closed-loop model error. The original physical certificate and
0.0025 gain floor remain; the reserve is an extra development predicate.
"""
import numpy as np
from experiments.joint_admissibility.gate import JointAdmissibilityGate


def variation_reserve(previous,baseline,candidate):
    p,b,u=(np.asarray(v,dtype=float) for v in (previous,baseline,candidate))
    if p.shape!=(3,) or b.shape!=(3,) or u.shape!=(3,):
        raise ValueError('Three finite per-car commands required')
    if not all(np.isfinite(v).all() for v in (p,b,u)):
        raise ValueError('Three finite per-car commands required')
    return .05*np.maximum(0.,np.abs(u-p)+np.abs(b-u)-np.abs(b-p))


class SwitchingReserveGate(JointAdmissibilityGate):
    def select(self,state,previous,unused=None):
        self._reserve_predictions={}
        command,info=super().select(state,previous,unused)
        selected=[r['switching_reserve'] for r in info['history_candidate_evidence'] if r['passed']]
        info.update(development_controller='switching_reserve_joint_admissibility',
                    switching_variation_cost_weight=.05,
                    selected_reserve_gain_score=selected[-1]['worst_reserved_relative_gain'] if selected else None,
                    local_variation_identity_is_global_error_bound=False)
        return command,info

    def predict(self,state,previous,mlo,mhi,command,path):
        value=super().predict(state,previous,mlo,mhi,command,path)
        key=(None if command is None else np.asarray(command,dtype=float).tobytes(),path)
        self._reserve_predictions[key]=value
        return value

    def _history_admissible(self,x,previous,mlo,mhi,command):
        history_ok,evidence=super()._history_admissible(x,previous,mlo,mhi,command)
        evidence['history_passed_before_reserve']=history_ok
        if not history_ok:
            evidence['switching_reserve']=dict(checked=False,reason='History predicate already rejected')
            return False,evidence
        # The baseline prediction was computed by the original selector.
        # Recover its actual command from the public baseline object, without
        # advancing the baseline or either observer a second time.
        baseline=self._current_public_baseline.copy()
        reserve=variation_reserve(previous,baseline,command)
        differences=[];costs=[]
        key=np.asarray(command,dtype=float).tobytes()
        for path in range(len(self.tapes)):
            before=self._reserve_predictions[None,path]
            after=self._reserve_predictions[key,path]
            differences.append((after[0]-before[0])[:,0])
            costs.append(float(before[0][:,0].mean()))
        delta=np.asarray(differences);costs=np.asarray(costs)
        scores=-(delta+reserve).mean(axis=1)/costs
        passed=bool(np.isfinite(scores).all() and np.all(delta+reserve<=1e-10)
                    and scores.min()>=self.gain_floor)
        evidence['switching_reserve']=dict(checked=True,passed=passed,
            previous=np.asarray(previous).tolist(),baseline=baseline.tolist(),
            candidate=np.asarray(command).tolist(),variation_reserve=reserve.tolist(),
            cost_differences=delta.tolist(),baseline_team_costs=costs.tolist(),
            worst_reserved_relative_gain=float(scores.min()),cost_weight=.05,
            local_identity_only=True,full_closed_loop_error_bound=False)
        evidence['passed']=passed
        return passed,evidence

    def _select(self,state,previous,unused=None):
        # Capture the one actual baseline selection in the ORIGINAL _select.
        # The scoped proxy forwards every attribute and never calls select twice.
        original=self.baseline;parent=self
        class CaptureBaseline:
            def __getattr__(self,name):return getattr(original,name)
            def select(self,*args):
                command,info=original.select(*args)
                parent._current_public_baseline=np.asarray(command).copy()
                return command,info
        self.baseline=CaptureBaseline()
        try:return super()._select(state,previous,unused)
        finally:self.baseline=original
