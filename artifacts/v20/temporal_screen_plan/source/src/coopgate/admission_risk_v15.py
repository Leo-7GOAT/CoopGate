"""Validate risk AMONG admitted scenes for a finite, previously frozen gate grid.

Uses one-sided exact binomial bounds and Bonferroni correction. This standard
learn-then-test construction is not a new statistical theorem. Validation must
be IID and separate from training/conformal fitting and the final performance
test. Fixed quotas from different regimes require stratum-specific validation
and a confidence budget across strata; independence alone is insufficient.
"""
import numpy as np
from scipy.stats import beta


def binomial_upper(harmful,admitted,delta):
    if not (0<delta<1) or admitted<0 or harmful<0 or harmful>admitted:
        raise ValueError('Invalid binomial counts or confidence level')
    if int(harmful)!=harmful or int(admitted)!=admitted:raise ValueError('Integer independent-scene counts required')
    if admitted==0 or harmful==admitted:return 1.
    return float(beta.ppf(1-delta,harmful+1,admitted-harmful))


def validate_gate_family(initial_ids,gate_ids,admitted,harmful,net_gain,
                         risk_cap=.05,delta=.05):
    """Select a certified gate; None means use the baseline for every scene.

    Arrays have shape (independent initial scenes, frozen candidate gates).
    harmful is the worst-mode/vehicle/metric performance-regression indicator
    within each initial scene. net_gain includes zero gain on rejected scenes.
    IDs bind complete fixed gates, including models, quantiles and thresholds.
    """
    ids=tuple(initial_ids);gates=tuple(gate_ids)
    a=np.asarray(admitted);h=np.asarray(harmful);gain=np.asarray(net_gain,dtype=float)
    if not ids or len(set(ids))!=len(ids):raise ValueError('Unique independent initial IDs required')
    if not gates or len(set(gates))!=len(gates):raise ValueError('Unique frozen gate IDs required')
    shape=(len(ids),len(gates))
    if a.shape!=shape or h.shape!=shape or gain.shape!=shape:raise ValueError('Expected initial x gate arrays')
    if a.dtype!=np.bool_ or h.dtype!=np.bool_:raise ValueError('Admission/harm flags must be boolean')
    if not np.isfinite(gain).all() or np.any(gain[~a]!=0):raise ValueError('Invalid gated gain accounting')
    if not (0<risk_cap<1 and 0<delta<1):raise ValueError('Invalid risk/confidence levels')
    candidates=[]
    for j,ident in enumerate(gates):
        n=int(a[:,j].sum());k=int((a[:,j]&h[:,j]).sum())
        upper=binomial_upper(k,n,delta/len(gates))
        candidates.append(dict(gate_id=ident,admitted=n,harmful=k,coverage=n/len(ids),
            conditional_empirical_risk=k/n if n else None,all_scene_harm_rate=k/len(ids),
            conditional_risk_upper=upper,certified=bool(n>0 and upper<=risk_cap),
            mean_net_gain=float(gain[:,j].mean())))
    eligible=[j for j,r in enumerate(candidates) if r['certified'] and r['mean_net_gain']>0.]
    selected=max(eligible,key=lambda j:(candidates[j]['mean_net_gain'],candidates[j]['admitted'],-j)) if eligible else None
    return dict(initials=len(ids),gates=candidates,selected_index=selected,
        selected_gate=gates[selected] if selected is not None else None,risk_cap=risk_cap,delta=delta,
        multiplicity=len(gates),statement='With probability at least 1-delta over IID validation scenes, every certified frozen gate has population harm risk among admitted scenes <= risk_cap; conditional on fixed training/calibration and gate family')
