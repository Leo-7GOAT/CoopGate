"""Selective-risk sensitivity on the three-outcome event space.

These deterministic probability inequalities do not estimate unknown shift
probabilities or issue a statistical certificate from development data.
"""
import math
from numbers import Real

import numpy as np


def probability(value, name, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or not 0 <= value <= 1
            or (positive and value == 0)):
        raise ValueError(name + ' must be a finite probability' + (' > 0' if positive else ''))
    return float(value)


def transfer_bound(source_risk_upper, source_adoption_lower, event_tv_upper):
    """For Q(A)>0, r_Q <= min(1, r_P + TV(P_Y,Q_Y)/P(A)).

    Y has labels: 0=no adoption, 1=adopted/non-harmful, 2=adopted/harmful.
    A positive target adoption probability follows from a_lower > tv_upper.
    Otherwise the risk bound is conditional on separately establishing Q(A)>0.
    Valid inputs are supplied probability bounds, never empirical certification.
    """
    risk = probability(source_risk_upper, 'Risk upper bound')
    adoption = probability(source_adoption_lower, 'Adoption lower bound', positive=True)
    shift = probability(event_tv_upper, 'Event-distribution TV upper bound')
    return dict(conditional_risk_upper=min(1., risk + shift/adoption),
        target_adoption_lower=max(0., adoption-shift),
        positive_target_adoption_guaranteed=adoption > shift)


def empirical_coupling(source, target):
    """Exact finite-table diagnostics; no population or confidence claim."""
    source, target = np.asarray(source), np.asarray(target)
    if (source.ndim != 1 or target.shape != source.shape or len(source) == 0
            or source.dtype.kind not in 'iu' or target.dtype.kind not in 'iu'
            or np.any(source > 2) or np.any(source < 0)
            or np.any(target > 2) or np.any(target < 0)):
        raise ValueError('Matched nonempty integer label arrays in {0,1,2} required')
    matrix = np.zeros((3,3), dtype=int)
    np.add.at(matrix, (source, target), 1)
    n = int(matrix.sum())
    p, q = matrix.sum(1)/n, matrix.sum(0)/n
    a, b = float(p[1:].sum()), float(q[1:].sum())
    risk_source = float(p[2]/a) if a > 0 else None
    risk_target = float(q[2]/b) if b > 0 else None
    tv = float(.5*np.abs(p-q).sum())
    mismatch = float(1-np.trace(matrix)/n)
    tv_bound = transfer_bound(risk_source,a,tv) if a > 0 else None
    coupling_bound = transfer_bound(risk_source,a,mismatch) if a > 0 else None
    if tv > mismatch+1e-12:
        raise AssertionError('Coupling cannot understate empirical total variation')
    if tv_bound is not None and risk_target is not None:
        if risk_target > tv_bound['conditional_risk_upper']+1e-12:
            raise AssertionError('Finite-table selective-risk inequality violated')
    return dict(paired_initials=n, transitions=matrix.tolist(),
        source_counts=matrix.sum(1).tolist(),target_counts=matrix.sum(0).tolist(),
        source_adoption=a,target_adoption=b,source_risk=risk_source,target_risk=risk_target,
        event_total_variation=tv,coupled_event_mismatch=mismatch,
        empirical_tv_envelope=tv_bound,empirical_coupling_envelope=coupling_bound,
        distribution_risk_certified=False)
