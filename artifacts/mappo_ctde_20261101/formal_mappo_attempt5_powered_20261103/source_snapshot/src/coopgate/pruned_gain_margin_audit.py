"""Independent arithmetic of the confidence prefix rejection witness."""
import numpy as np
from .gain_margin_audit import review_gain_trace,validate_gain_evidence


def validate_prefix_evidence(item):
    validate_gain_evidence(item)
    if item['confidence_pruning_kind']!='rejection_only_min_over_paths_upper_bound':raise ValueError('Wrong pruning identity')
    evidence={e['candidate']:e for e in item['candidate_evidence']}
    seen=set()
    for proof in item['gain_margin_prefix_rejections']:
        candidate=proof['candidate'];path=proof['path'];record=evidence[candidate]
        if candidate in seen or record['passed'] or record['checked_paths']!=path+1 or len(record['differences'])!=path+1:
            raise ValueError('Prefix rejection path identity differs')
        seen.add(candidate);base=proof['baseline_team_cost'];floor=item['gain_confidence_floor']
        if floor<=0 or proof['floor']!=floor:raise ValueError('Zero floor must not prune')
        if np.isfinite(base) and base>0:
            d=record['differences'][path];upper=-(d[0][0]+d[1][0]+d[2][0])/3./base
            if abs(upper-proof['score_upper_bound'])>1e-12 or not upper<floor:
                raise ValueError('Prefix is not a valid below-floor upper bound')
        elif proof['score_upper_bound'] is not None:raise ValueError('Invalid baseline denominator not refused')


def review_pruned_gain_trace(row,trace):
    proof=review_gain_trace(row,trace)
    if row['rule']!='baseline':
        for item in trace:validate_prefix_evidence(item)
    return proof
