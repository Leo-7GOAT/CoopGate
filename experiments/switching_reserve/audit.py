"""Independently check added variation reserves without rewriting raw evidence."""
from copy import deepcopy
from types import FunctionType

import numpy as np
from experiments.joint_admissibility_recovery import audit as previous


def history_passed(r,floor):
    if not r['fitted_forecast_available']:return True
    delta=np.asarray(r['differences'],dtype=float);phase=np.asarray(r['phase_differences'],dtype=float)
    if delta.shape!=(3,2) or phase.shape!=(3,):raise ValueError('History dimensions differ')
    score=r['relative_gain']
    return bool(r['complete_pair'] and np.isfinite(delta).all() and np.isfinite(phase).all()
                and score is not None and np.isfinite(score) and score>=floor
                and np.all(delta<=1e-10) and np.all(phase<=1e-10))


def validate(item):
    if item.get('development_controller')!='switching_reserve_joint_admissibility':
        return previous.validate_joint_trace(item)
    histories=item['history_candidate_evidence'];extra_vetoes=set()
    for r in histories:
        before=history_passed(r,item['gain_confidence_floor']);z=r['switching_reserve']
        if r['history_passed_before_reserve']!=before:raise ValueError('Original history predicate differs')
        if not before:
            if z['checked'] or r['passed']:raise ValueError('Already rejected history must remain rejected')
            continue
        if not z['checked'] or z['cost_weight']!=.05:raise ValueError('Variation reserve must be checked')
        if not np.array_equal(z['baseline'],item['baseline_command']):raise ValueError('Backup command differs')
        p,b,u=(np.asarray(z[k],dtype=float) for k in ('previous','baseline','candidate'))
        if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (p,b,u)):
            raise ValueError('Finite per-car reserve commands required')
        # Independent distance-to-segment form of the triangle-excess identity.
        charge=.1*(np.maximum(np.minimum(p,b)-u,0.)+np.maximum(u-np.maximum(p,b),0.))
        if not np.allclose(charge,z['variation_reserve'],rtol=0,atol=1e-12):
            raise ValueError('Control-variation reserve arithmetic differs')
        delta=np.asarray(z['cost_differences'],dtype=float);base=np.asarray(z['baseline_team_costs'],dtype=float)
        if (delta.shape!=(item['forecast_bank_size'],3) or base.shape!=(item['forecast_bank_size'],)
                or not np.isfinite(delta).all() or not np.isfinite(base).all() or np.any(base<=0)):
            raise ValueError('Complete finite reserve forecasts required')
        score=float((-(delta+charge).mean(axis=1)/base).min())
        passed=bool(np.all(delta+charge<=1e-10) and score>=item['gain_confidence_floor'])
        if abs(score-z['worst_reserved_relative_gain'])>1e-12 or z['passed']!=passed or r['passed']!=passed:
            raise ValueError('Reserved gain admission differs')
        if not passed:extra_vetoes.add(r['candidate'])
    # Independently validate gain evidence that will be excluded from the old
    # final-admission accounting. Its rejection is a reserve veto, not a fake
    # history failure. All original histories remain unmodified in raw storage.
    candidates={r['candidate']:r for r in item['candidate_evidence']}
    margins={r['candidate']:r for r in item['gain_margin_evidence']}
    for index in extra_vetoes:
        c=candidates[index];m=margins[index]
        diff=np.asarray(c['differences'],dtype=float);base=np.asarray(m['baseline_team_costs'],dtype=float)
        if (diff.shape!=(item['forecast_bank_size'],3,2) or base.shape!=(item['forecast_bank_size'],)
                or not np.isfinite(diff).all() or not np.isfinite(base).all() or np.any(base<=0)):
            raise ValueError('Complete gain evidence required for reserve veto')
        score=float((-diff[:,:,0].mean(axis=1)/base).min())
        if (abs(score-m['worst_relative_team_gain'])>1e-12 or score<item['gain_confidence_floor']
                or not m['passed'] or c['passed'] or not c.get('history_rejected')):
            raise ValueError('Reserve-vetoed gain evidence differs')
    projection=deepcopy(item);projection['development_controller']='joint_candidate_admissibility'
    projection['gain_margin_evidence']=[r for r in projection['gain_margin_evidence'] if r['candidate'] not in extra_vetoes]
    projection['history_candidate_evidence']=[r for r in projection['history_candidate_evidence'] if r['candidate'] not in extra_vetoes]
    previous.validate_joint_trace(projection)
    selected=[r for r in histories if r['passed']]
    score=selected[-1]['switching_reserve']['worst_reserved_relative_gain'] if selected else None
    if item['selected_reserve_gain_score']!=score:raise ValueError('Actual selected reserve confidence differs')
    if len(selected)!=int(item['actual_learning_used']):raise ValueError('Only the actual selected candidate may pass')


_scope=dict(previous.original.__dict__);_scope['validate_prefix_evidence']=validate
_replay=FunctionType(previous.original.audit_record.__code__,_scope,
    name='independent_switching_reserve_replay',argdefs=previous.original.audit_record.__defaults__)
_replay.__kwdefaults__=previous.original.audit_record.__kwdefaults__.copy()


def audit_record(case,record,model,*,reference_factory):
    return _replay(case,record,model,reference_factory=reference_factory)
