"""Uniform conditional-risk envelopes for changes in known stratum weights.

Only mixture proportions may change. The within-stratum laws, frozen policy,
and outcome definitions must remain unchanged. This is not an OOD certificate.
"""
from fractions import Fraction
from itertools import product

import numpy as np

from .stratified_risk import conditional_risk_upper


def _vector(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not 1 <= len(result) <= 8 or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite vector with 1..8 strata')
    if np.any(result < 0.) or np.any(result > 1.):
        raise ValueError(name+' must lie in [0,1]')
    return result


def _fractions(values):
    return tuple(Fraction(float(x)) for x in values)


def _vertices(lower, upper):
    low, high = _fractions(lower), _fractions(upper)
    if len(low) != len(high) or any(a > b for a,b in zip(low,high)):
        raise ValueError('Mixture box bounds differ in length or order')
    if sum(low) > 1 or sum(high) < 1:
        raise ValueError('Mixture box has no probability vector')
    result=set()
    for free in range(len(low)):
        fixed=[j for j in range(len(low)) if j != free]
        for endpoints in product((0,1),repeat=len(fixed)):
            vertex=list(low)
            for j, endpoint in zip(fixed,endpoints):
                vertex[j]=high[j] if endpoint else low[j]
            vertex[free]=1-sum(vertex[j] for j in fixed)
            if low[free] <= vertex[free] <= high[free]:
                result.add(tuple(vertex))
    if not result:
        raise ValueError('Mixture box has no vertices')
    return sorted(result)


def _outward(value, upper):
    rounded=float(value)
    if (upper and Fraction(rounded) < value) or (not upper and Fraction(rounded) > value):
        rounded=float(np.nextafter(rounded,np.inf if upper else -np.inf))
    return rounded


def mixture_box_envelope(joint_harm_uppers, adoption_lowers, weight_lower, weight_upper):
    """Optimize already-simultaneous bounds over a box intersected with simplex.

    Vertex coordinates and sums use rational representations of input floats.
    This makes optimization exact for those supplied numerical bounds; it does
    not interval-verify the upstream floating-point beta quantile computation.
    """
    q=_vector(joint_harm_uppers,'Joint harm bounds')
    a=_vector(adoption_lowers,'Adoption bounds')
    low=_vector(weight_lower,'Weight lower bounds')
    high=_vector(weight_upper,'Weight upper bounds')
    if not len(q)==len(a)==len(low)==len(high):
        raise ValueError('All stratum vectors must have the same length')
    vertices=_vertices(low,high)
    q,a=_fractions(q),_fractions(a)
    denominators=[sum(wj*aj for wj,aj in zip(w,a)) for w in vertices]
    adoption_index=min(range(len(vertices)),key=lambda i:denominators[i])
    ratios=[]
    for w, denominator in zip(vertices,denominators):
        numerator=sum(wj*qj for wj,qj in zip(w,q))
        ratios.append(min(Fraction(1),numerator/denominator) if denominator > 0 else Fraction(1))
    risk_index=max(range(len(vertices)),key=lambda i:ratios[i])
    def witness(w):
        return dict(weights=[float(x) for x in w],exact_weights=[str(x) for x in w])
    return dict(risk_upper=_outward(ratios[risk_index],True),
        adoption_lower=_outward(denominators[adoption_index],False),
        risk_witness=witness(vertices[risk_index]),
        adoption_witness=witness(vertices[adoption_index]),vertices_checked=len(vertices),
        zero_denominator_feasible=denominators[adoption_index]==0,
        weight_lower=low.tolist(),weight_upper=high.tolist(),
        optimization_exact_for_supplied_binary_float_bounds=True)


def conditional_risk_under_mixture_shift(harmful_adoptions, adoptions, totals,
                                       weight_lower, weight_upper, *, alpha,
                                       risk_limit=.05, adoption_min=.20):
    low,high=_vector(weight_lower,'Weight lower bounds'),_vector(weight_upper,'Weight upper bounds')
    _vertices(low,high)
    if not np.isfinite(risk_limit) or not 0 < risk_limit < 1:
        raise ValueError('risk_limit must lie in (0,1)')
    if not np.isfinite(adoption_min) or not 0 < adoption_min <= 1:
        raise ValueError('adoption_min must lie in (0,1]')
    # Stratum CP bounds do not depend on weights. The frozen routine requires
    # strictly positive weights, so use uniform auxiliary weights and discard
    # its aggregate. These auxiliary weights need not lie in the requested box.
    fixed=conditional_risk_upper(harmful_adoptions,adoptions,totals,
                                [1./len(low)]*len(low),alpha=alpha)
    envelope=mixture_box_envelope(fixed['stratum_joint_harm_uppers'],
        fixed['stratum_adoption_lowers'],low,high)
    return dict(**envelope,stratum_joint_harm_uppers=fixed['stratum_joint_harm_uppers'],
        stratum_adoption_lowers=fixed['stratum_adoption_lowers'],allocated_error=float(alpha),
        component_alpha=fixed['component_alpha'],risk_limit=float(risk_limit),
        adoption_min=float(adoption_min),
        risk_and_adoption_bounds_pass=bool(envelope['risk_upper'] <= risk_limit
                                          and envelope['adoption_lower'] >= adoption_min),
        simultaneous_over_all_weights_in_box=True,
        gain_certificate_transferred=False,joint_deployment_certified=False,
        within_stratum_distribution_change_covered=False,
        physical_safety_certified=False)
