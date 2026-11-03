"""Validate every veto before applying the unchanged physical replay.

Historical raw traces are never edited. A temporary in-memory projection lets
the old single-admission checker validate the final admission and prefix bounds.
Every additionally vetoed candidate receives separate gain/history checks.
The frozen physics auditor runs its exact original code with this local
validator binding; no module global or historical production source is patched.
"""
from copy import deepcopy
from types import FunctionType

import numpy as np
from coopgate.pruned_gain_margin_audit import validate_prefix_evidence
import mappo_coupled_formal_runtime as original


def validate_joint_trace(item):
    if item.get('development_controller')!='joint_candidate_admissibility':
        return validate_prefix_evidence(item)
    histories=item['history_candidate_evidence']
    history={r['candidate']:r for r in histories}
    if len(history)!=len(histories):raise ValueError('Duplicate candidate history evidence')
    candidates={r['candidate']:r for r in item['candidate_evidence']}
    margins={r['candidate']:r for r in item['gain_margin_evidence']}
    if len(margins)!=len(item['gain_margin_evidence']):raise ValueError('Duplicate gain evidence')
    accepted_gain={k for k,r in margins.items() if r['passed']}
    if set(history)!=accepted_gain:raise ValueError('Every gain-qualified candidate needs its history check')
    rejected=set()
    for index,r in history.items():
        if not r['fitted_forecast_available']:
            passed=True
        else:
            delta=np.asarray(r['differences'],dtype=float)
            phase=np.asarray(r['phase_differences'],dtype=float)
            if delta.shape!=(3,2) or phase.shape!=(3,):
                raise ValueError('Incomplete history forecast evidence')
            score=r['relative_gain']
            passed=bool(r['complete_pair'] and np.isfinite(delta).all() and
                        np.isfinite(phase).all() and score is not None and
                        np.isfinite(score) and score>=item['gain_confidence_floor'] and
                        np.all(delta<=1e-10) and np.all(phase<=1e-10))
        if r['passed']!=passed:raise ValueError('History admissibility differs')
        if not passed:
            rejected.add(index)
            e=candidates[index];m=margins[index]
            if e['passed'] or not e.get('history_rejected'):
                raise ValueError('History veto must reject this candidate')
            diff=np.asarray(e['differences'],dtype=float)
            base=np.asarray(m['baseline_team_costs'],dtype=float)
            if (diff.shape!=(item['forecast_bank_size'],3,2) or
                    base.shape!=(item['forecast_bank_size'],) or
                    not np.isfinite(diff).all() or not np.isfinite(base).all() or
                    np.any(base<=0)):
                raise ValueError('Complete finite bank required for a history veto')
            scores=-diff[:,:,0].mean(axis=1)/base
            if (abs(float(scores.min())-m['worst_relative_team_gain'])>1e-12 or
                    scores.min()<item['gain_confidence_floor'] or
                    m['floor']!=item['gain_confidence_floor'] or not m['passed']):
                raise ValueError('Vetoed candidate gain arithmetic differs')
    projected=deepcopy(item)
    projected['gain_margin_evidence']=[r for r in projected['gain_margin_evidence']
                                       if r['candidate'] not in rejected]
    # This checks the actual final selection and original prefix arithmetic.
    validate_prefix_evidence(projected)
    if item['actual_learning_used']:
        phase=np.asarray(item['selected_phase_differences'],dtype=float)
        if not np.isfinite(phase).all() or np.any(phase>1e-10):
            raise ValueError('Final selection violates strict crossing phase')
    return dict(history_candidates=len(history),history_vetoes=len(rejected))


_scope=dict(original.__dict__)
_scope['validate_prefix_evidence']=validate_joint_trace
_physical_replay=FunctionType(original.audit_record.__code__,_scope,
    name='independent_joint_trace_physical_replay',argdefs=original.audit_record.__defaults__)
_physical_replay.__kwdefaults__=original.audit_record.__kwdefaults__.copy()


def audit_record(case,record,model,*,reference_factory):
    """Full controller + actual physical reconstruction, not just log arithmetic."""
    return _physical_replay(case,record,model,reference_factory=reference_factory)
