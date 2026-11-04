"""Reconstruct the new score independently of its runtime helper."""
import numpy as np
from .parameter_gate_audit import review_parameter_trace


def validate_gain_evidence(item):
    floor=item['gain_confidence_floor']
    if floor not in (0.,.0025,.005,.01) or item['gain_score_is_probability'] or item['performance_confidence_kind']!='normalized_worst_path_team_gain':
        raise ValueError('Unregistered gain confidence contract')
    candidates={e['candidate']:e for e in item['candidate_evidence']};passed=[]
    for entry in item['gain_margin_evidence']:
        evidence=candidates[entry['candidate']];base=entry['baseline_team_costs'];diff=evidence['differences']
        if len(base)!=item['forecast_bank_size'] or len(diff)!=len(base) or any(not np.isfinite(b) or b<=0 for b in base):
            raise ValueError('Complete positive baseline cost denominator required')
        scores=[-sum(d[i][0] for i in range(3))/3./b for d,b in zip(diff,base)]
        score=min(scores)
        if not np.isfinite(score) or abs(score-entry['worst_relative_team_gain'])>1e-12:
            raise ValueError('Relative gain confidence score differs')
        accept=floor==0 or score>=floor
        if entry['passed']!=accept or entry['floor']!=floor or evidence['passed']!=accept:
            raise ValueError('Gain threshold admission differs')
        if accept:passed.append(entry)
    used=item['actual_learning_used']
    if used:
        if len(passed)!=1 or passed[0] is not item['gain_margin_evidence'][-1]:raise ValueError('Selected evidence order differs')
        selected=passed[0]
        if (selected['composition']!=item['composition'] or selected['repair_weight']!=item['repair_weight']
                or selected['baseline_team_costs']!=item['selected_baseline_forecast_team_costs']
                or selected['worst_relative_team_gain']!=item['selected_gain_confidence_score']):
            raise ValueError('Selected gain confidence identity differs')
    elif passed or item['selected_gain_confidence_score'] is not None or item['selected_baseline_forecast_team_costs']:
        raise ValueError('Fallback falsely claims an accepted gain confidence score')


def review_gain_trace(row,trace):
    proof=review_parameter_trace(row,trace,None)
    if row['rule']!='baseline':
        for item in trace:validate_gain_evidence(item)
    return proof
