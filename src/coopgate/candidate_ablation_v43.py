"""A 2x2 candidate-family x performance-path mechanism ablation.

Reuse the *identical* frozen selector function code with an isolated globals
mapping for raw compositions. Nothing is patched in the imported V35 module.
Within each candidate family, nominal/bank selection differs only in V35's
existing path-count branch. These are internal ablations, not literature methods.
"""
from types import FunctionType
import numpy as np

from .actuator_recovery_v34 import stopping_certificate
from .joint_risk_data_v42 import episode_outcome
from .recovery_joint_gate_v35 import RecoveryJointGate


VARIANTS = ('baseline', 'raw_nominal', 'raw_bank', 'projected_nominal', 'projected_bank')


def raw_compositions(state, previous, proposal, baseline, mlo, mhi, bounds, packed):
    """Only physically admissible raw learned/baseline masks, without repair."""
    candidates, seen = [], [baseline.copy()]
    for code in range(1, 8):
        mask = np.array([bool(code & (1 << i)) for i in range(3)])
        command = np.where(mask, proposal, baseline)
        command[state[:, 0] >= packed[11]] = baseline[state[:, 0] >= packed[11]]
        if any(np.array_equal(command, prior) for prior in seen):
            continue
        if not stopping_certificate(state, previous, command, mlo, mhi, bounds, packed)[0]:
            continue
        seen.append(command.copy())
        candidates.append((code, 1., command.copy(), -2))  # -2: no projection performed
    return candidates, len(candidates)


# Frozen V35 has no constructor-level candidate-generator injection. FunctionType
# supplies that dependency without source duplication, editing frozen files, or
# mutating module globals. All other function globals and the code object agree.
_RAW_SELECT = FunctionType(RecoveryJointGate.select.__code__,
    {**RecoveryJointGate.select.__globals__, 'compositions': raw_compositions},
    name='raw_candidate_select', argdefs=RecoveryJointGate.select.__defaults__,
    closure=RecoveryJointGate.select.__closure__)


def rule_for(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown fixed factorial variant')
    if variant == 'baseline':
        return 'baseline'
    return 'nominal_performance' if variant.endswith('_nominal') else 'bank_performance'


class CandidateAblationGate(RecoveryJointGate):
    def __init__(self, model, profile, variant):
        self.variant = variant
        super().__init__(model, profile, rule_for(variant))

    def select(self, state, previous, unused=None):
        select = _RAW_SELECT if self.variant.startswith('raw_') else RecoveryJointGate.select
        command, info = select(self, state, previous, unused)
        info.update(candidate_ablation_variant=self.variant,
                    candidate_family=('baseline' if self.variant == 'baseline' else
                                      'raw' if self.variant.startswith('raw_') else 'projected'),
                    mechanism_ablation_only=True)
        return command, info


def paired_mechanism_summary(cluster_ids, records):
    """Complete paired descriptive effects; no significance or risk certificate.

    All clusters stay in every contrast, including capped costs of failed runs.
    Joint harm is H AND A over all clusters, not a difference between selected
    adopted subsets. Report adoption alongside it to expose abstention tradeoffs.
    Callers must verify sealed records and physical/selector replay first.
    """
    expected = {(cluster, variant) for cluster in cluster_ids for variant in VARIANTS}
    if not cluster_ids or len(expected) != 5*len(cluster_ids):
        raise ValueError('Distinct complete independent-cluster identities required')
    rows = {}
    for record in records:
        key = (record['cluster_id'], record['variant'])
        if key in rows or key not in expected:
            raise ValueError('Duplicate or foreign factorial observation')
        rows[key] = episode_outcome(record['row'])
    if set(rows) != expected:
        raise ValueError('All five methods for every predeclared cluster are required')
    values = {name: [] for name in VARIANTS[1:]}
    baseline_failures = 0
    for cluster in cluster_ids:
        base = rows[cluster, 'baseline']
        if base['adopted']:
            raise ValueError('Baseline cannot adopt learning')
        baseline_failures += base['failed']
        for variant in VARIANTS[1:]:
            row = rows[cluster, variant]
            harm = bool(base['failed'] or row['failed'] or np.any(row['costs'] > base['costs']+1e-9)
                        or np.any(row['arrivals'] > base['arrivals']+1e-9))
            if not row['adopted'] and not base['failed'] and not row['failed']:
                if (not np.allclose(row['costs'], base['costs'], atol=1e-9, rtol=0)
                        or not np.allclose(row['arrivals'], base['arrivals'], atol=1e-9, rtol=0)):
                    raise ValueError('Successful zero-adoption trajectory differs from baseline')
            values[variant].append([100*(base['team_cost']-row['team_cost'])/base['team_cost'],
                                   float(harm and row['adopted']), float(row['adopted']), float(row['failed'])])
    arrays = {k: np.asarray(v) for k, v in values.items()}
    contrasts = {
        'bank_minus_nominal_with_raw_candidates': arrays['raw_bank']-arrays['raw_nominal'],
        'bank_minus_nominal_with_projected_candidates': arrays['projected_bank']-arrays['projected_nominal'],
        'projection_minus_raw_with_nominal_gate': arrays['projected_nominal']-arrays['raw_nominal'],
        'projection_minus_raw_with_bank_gate': arrays['projected_bank']-arrays['raw_bank'],
        'interaction': (arrays['projected_bank']-arrays['projected_nominal'])-(arrays['raw_bank']-arrays['raw_nominal']),
    }
    names = ('gain_percentage_points', 'joint_harm_probability', 'adoption_probability', 'failure_probability')
    return dict(clusters=len(cluster_ids), actual_sumo_runs=5*len(cluster_ids),
        baseline_failures=int(baseline_failures), failed_pairs_removed=0,
        method_means={k: dict(zip(names, v.mean(axis=0).tolist())) for k, v in arrays.items()},
        paired_contrasts={k: dict(zip(names, v.mean(axis=0).tolist())) for k, v in contrasts.items()},
        descriptive_only=True, statistical_risk_certified=False, standalone_latency_comparison=False,
        conditional_adopted_subsets_used_for_causal_contrasts=False)
