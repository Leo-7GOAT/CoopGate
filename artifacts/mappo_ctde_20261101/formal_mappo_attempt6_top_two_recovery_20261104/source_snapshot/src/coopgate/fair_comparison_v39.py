"""Pre-registered, equal-information comparison adapters for recovery gating.

The expensive evidence packet is constructed once under one fixed budget and
is then consumed by every decision rule.  Literature-derived rules are marked
as adapters unless their original system, optimizer, and theorem assumptions
are reproduced.  This module defines protocol and analysis logic only: it does
not read outcomes, create artifacts, or launch a simulator.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from numbers import Integral

import numpy as np
from scipy.stats import beta, t


METHODS = (
    'proposed_bank_performance_gate',
    'always_learned',
    'always_baseline',
    'uncalibrated_confidence_threshold',
    'existing_calibrated_gate',
    'advantage_lcb_adapter',
    'mps_adapter',
    'mamps_adapter',
)
CANDIDATE_METHODS = tuple(name for name in METHODS if name != 'always_baseline')
MASKS = tuple(tuple(bool(code & (1 << i)) for i in range(3)) for code in range(8))
COST_CONTRACT = ('capped per-vehicle arrival time + 0.05 command variation; '
                 'failures, collisions, aborts and timeouts are retained')
OBSERVATION_CONTRACT = ('same observed kinematics, executed previous command, decision index, '
                        'declared actuator class and command-history interval')


@dataclass(frozen=True)
class MethodDescriptor:
    name: str
    family: str
    decision_rule_status: str
    closed_loop_status: str
    scope: str


METHOD_DESCRIPTORS = (
    MethodDescriptor('proposed_bank_performance_gate', 'proposed_method',
        'project_full_reproduction', 'project_full_reproduction',
        'The bound V35 RecoveryJointGate bank_performance decision and recovery execution; '
        'its finite-bank margin is not a calibrated probability.'),
    MethodDescriptor('always_learned', 'negative_control', 'protocol_control',
        'protocol_control', 'Executes the one common learned proposal without admission.'),
    MethodDescriptor('always_baseline', 'negative_control', 'protocol_control',
        'project_full_reproduction', 'Executes the one common V35 recovery baseline.'),
    MethodDescriptor('uncalibrated_confidence_threshold', 'project_comparator',
        'full_reproduction', 'protocol_control',
        'Applies the frozen ordinary scalar confidence threshold, without a calibration claim.'),
    MethodDescriptor('existing_calibrated_gate', 'project_comparator',
        'full_reproduction', 'adapter',
        'Reproduces predicted_delta + calibration_quantile <= -minimum_gain; the common V35 '
        'stepwise recovery execution is an adapter to the earlier episode gate.'),
    MethodDescriptor('advantage_lcb_adapter', 'safe_policy_improvement', 'adapter', 'adapter',
        'A fixed-bank lower-confidence-bound rule; not an original CSPI estimator or theorem.'),
    MethodDescriptor('mps_adapter', 'model_predictive_shielding', 'adapter', 'adapter',
        'All-or-none switching with the V35 common physical oracle, not original MPS.'),
    MethodDescriptor('mamps_adapter', 'multi_agent_model_predictive_shielding', 'adapter', 'adapter',
        'Monotone per-agent switching with the V35 common physical oracle, not original MAMPS.'),
)
DESCRIPTORS = {item.name: item for item in METHOD_DESCRIPTORS}


def _finite(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(name + ' must be finite')
    return value


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return int(value)


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(name + ' must be a lowercase SHA-256 digest')
    return value


@dataclass(frozen=True)
class ComputeBudget:
    observation_reads: int = 1
    learned_proposal_calls: int = 1
    baseline_proposal_calls: int = 1
    forecast_paths: int = 33
    safety_mask_checks: int = 8
    proposed_gate_replays: int = 1
    maximum_proposed_gate_forecast_calls: int = 957
    decision_timeout_ms: float = 500.

    def __post_init__(self):
        for name in ('observation_reads', 'learned_proposal_calls', 'baseline_proposal_calls',
                     'forecast_paths', 'safety_mask_checks', 'proposed_gate_replays',
                     'maximum_proposed_gate_forecast_calls'):
            _positive_integer(getattr(self, name), name)
        if self.safety_mask_checks != len(MASKS):
            raise ValueError('The common budget must evaluate all eight three-agent masks')
        if self.forecast_paths < 2:
            raise ValueError('Advantage uncertainty requires at least two fixed forecast paths')
        if _finite(self.decision_timeout_ms, 'Decision timeout') <= 0:
            raise ValueError('Decision timeout must be positive')


@dataclass(frozen=True)
class ProtocolSpec:
    baseline_implementation_sha256: str
    proposed_gate_implementation_sha256: str
    calibration_sha256: str
    confidence_score_definition_sha256: str
    forecast_bank_sha256: str
    physical_oracle_sha256: str
    budget: ComputeBudget = field(default_factory=ComputeBudget)
    methods: tuple = METHODS
    uncalibrated_confidence_threshold: float = .8
    calibrated_minimum_gain_s: float = .05
    advantage_lcb_multiplier: float = 1.96
    advantage_min_gain_s: float = 0.
    adoption_minimum: float = .20
    gain_minimum_percent: float = .5
    family_error: float = .05
    risk_error_share: float = .50
    adoption_error_share: float = .25
    gain_error_share: float = .25
    cost_contract: str = COST_CONTRACT
    observation_contract: str = OBSERVATION_CONTRACT
    primary_metric: str = 'simultaneous_exact_upper_bound_on_cluster_harm_given_adoption'

    def __post_init__(self):
        for name in ('baseline_implementation_sha256', 'proposed_gate_implementation_sha256',
                     'calibration_sha256',
                     'confidence_score_definition_sha256', 'forecast_bank_sha256',
                     'physical_oracle_sha256'):
            _sha256(getattr(self, name), name)
        if tuple(self.methods) != METHODS:
            raise ValueError('The complete pre-registered comparator family is mandatory')
        if not 0. < _finite(self.uncalibrated_confidence_threshold,
                            'Uncalibrated threshold') < 1.:
            raise ValueError('Uncalibrated threshold must lie in (0,1)')
        if _finite(self.calibrated_minimum_gain_s, 'Calibrated minimum gain') < 0.:
            raise ValueError('Calibrated minimum gain must be nonnegative')
        if _finite(self.advantage_lcb_multiplier, 'Advantage LCB multiplier') <= 0.:
            raise ValueError('Advantage LCB multiplier must be positive')
        _finite(self.advantage_min_gain_s, 'Advantage minimum gain')
        for name in ('adoption_minimum', 'family_error'):
            value = _finite(getattr(self, name), name)
            if not 0. < value < 1.:
                raise ValueError(name + ' must lie in (0,1)')
        if _finite(self.gain_minimum_percent, 'Gain minimum') <= 0.:
            raise ValueError('Gain minimum must be positive')
        shares = np.asarray([self.risk_error_share, self.adoption_error_share,
                             self.gain_error_share], dtype=float)
        if (not np.isfinite(shares).all() or np.any(shares <= 0.)
                or not np.isclose(shares.sum(), 1., atol=1e-14, rtol=0)):
            raise ValueError('Risk, adoption and gain error shares must be positive and sum to one')
        if not self.cost_contract or not self.observation_contract or not self.primary_metric:
            raise ValueError('Cost, observation and primary-metric contracts are required')


@dataclass(frozen=True)
class CaseIdentity:
    case_id: str
    independent_cluster: str
    policy_id: str
    policy_sha256: str
    scenario: str
    initial_id: int
    dynamics: str
    disturbance: str
    initial_sha256: str
    dynamics_sha256: str
    disturbance_sha256: str

    def __post_init__(self):
        for name in ('case_id', 'independent_cluster', 'policy_id', 'scenario', 'dynamics',
                     'disturbance', 'initial_sha256', 'dynamics_sha256', 'disturbance_sha256'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(name + ' is required')
        if isinstance(self.initial_id, (bool, np.bool_)) or not isinstance(self.initial_id, Integral):
            raise ValueError('Initial ID must be an integer')
        for name in ('policy_sha256', 'initial_sha256', 'dynamics_sha256', 'disturbance_sha256'):
            _sha256(getattr(self, name), name)


@dataclass(frozen=True)
class WorkLedger:
    observation_reads: int
    learned_proposal_calls: int
    baseline_proposal_calls: int
    forecast_paths: int
    safety_mask_checks: int
    proposed_gate_replays: int
    proposed_gate_forecast_calls: int
    elapsed_ms: float
    timed_out: bool = False


@dataclass(frozen=True)
class DecisionEvidence:
    case: CaseIdentity
    step: int
    observation_fingerprint: str
    proposal: tuple
    baseline: tuple
    active: tuple
    confidence: float
    predicted_delta_s: float
    calibration_quantile_s: float
    in_calibration_domain: bool
    prediction_complete: bool
    advantage_samples_s: tuple
    safety_flags_by_mask: tuple
    proposed_bank_command: tuple
    proposed_bank_admitted: bool
    proposed_bank_physical_admissible: bool
    proposed_bank_prediction_complete: bool
    proposed_bank_replay_sha256: str
    proposed_bank_candidate_count: int
    ledger: WorkLedger


@dataclass(frozen=True)
class Decision:
    method: str
    command: tuple
    learning_mask: tuple
    adopted: bool
    physical_admissible: bool
    reason: str
    evidence_sha256: str
    decision_rule_status: str
    closed_loop_status: str
    score: float | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class Assignment:
    case: CaseIdentity
    method: str
    order: int


@dataclass(frozen=True)
class EpisodeRecord:
    case: CaseIdentity
    method: str
    protocol_sha256: str
    cost_contract: str
    timeout_ms: float
    adopted: bool
    completed: bool
    timed_out: bool
    collision: bool
    aborted: bool
    contract_violations: int
    per_vehicle_cost: tuple
    per_vehicle_arrival: tuple
    team_cost: float


def _jsonable(value):
    if hasattr(value, '__dataclass_fields__'):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def fingerprint(value):
    payload = _jsonable(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _mask_code(mask):
    return sum((1 << i) for i, value in enumerate(mask) if value)


def _command(evidence, mask):
    return tuple(evidence.proposal[i] if mask[i] else evidence.baseline[i] for i in range(3))


def _validate_ledger(ledger, budget):
    values = ('observation_reads', 'learned_proposal_calls', 'baseline_proposal_calls',
              'forecast_paths', 'safety_mask_checks', 'proposed_gate_replays')
    for name in values:
        value, maximum = getattr(ledger, name), getattr(budget, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value <= maximum:
            raise ValueError('Invalid common work ledger: '+name)
        if not ledger.timed_out and value != maximum:
            raise ValueError('Every completed method packet must consume the identical full budget')
    calls = ledger.proposed_gate_forecast_calls
    if (isinstance(calls, (bool, np.bool_)) or not isinstance(calls, Integral)
            or not 0 <= calls <= budget.maximum_proposed_gate_forecast_calls):
        raise ValueError('Invalid proposed-gate forecast-call count')
    if _finite(ledger.elapsed_ms, 'Common evidence elapsed time') < 0.:
        raise ValueError('Common evidence elapsed time cannot be negative')
    if not isinstance(ledger.timed_out, (bool, np.bool_)):
        raise ValueError('Timed-out flag must be Boolean')
    if not ledger.timed_out and ledger.elapsed_ms > budget.decision_timeout_ms + 1e-12:
        raise ValueError('A packet over the common deadline must be marked timed out')
    if ledger.timed_out and ledger.elapsed_ms < budget.decision_timeout_ms - 1e-12:
        raise ValueError('A common timeout requires reaching the common deadline')


def validate_evidence(evidence, spec):
    if isinstance(evidence.step, (bool, np.bool_)) or not isinstance(evidence.step, Integral) or evidence.step < 0:
        raise ValueError('Decision step must be a nonnegative integer')
    _sha256(evidence.observation_fingerprint, 'Common observation fingerprint')
    proposal, baseline = np.asarray(evidence.proposal, dtype=float), np.asarray(evidence.baseline, dtype=float)
    if (proposal.shape != (3,) or baseline.shape != (3,) or
            not np.isfinite(proposal).all() or not np.isfinite(baseline).all()):
        raise ValueError('One finite common three-agent proposal and baseline are required')
    active = np.asarray(evidence.active)
    if active.shape != (3,) or active.dtype != np.dtype(bool):
        raise ValueError('Three explicit active-agent flags required')
    if not np.array_equal(proposal[~active], baseline[~active]):
        raise ValueError('Inactive agents must be pinned to the common baseline')
    confidence = _finite(evidence.confidence, 'Confidence')
    if not 0. <= confidence <= 1.:
        raise ValueError('Confidence must lie in [0,1]')
    _finite(evidence.predicted_delta_s, 'Predicted cost difference')
    _finite(evidence.calibration_quantile_s, 'Calibration quantile')
    if not isinstance(evidence.in_calibration_domain, (bool, np.bool_)):
        raise ValueError('Calibration-domain flag must be Boolean')
    if not isinstance(evidence.prediction_complete, (bool, np.bool_)):
        raise ValueError('Prediction-complete flag must be Boolean')
    advantage = np.asarray(evidence.advantage_samples_s, dtype=float)
    if advantage.shape != (spec.budget.forecast_paths,) or not np.isfinite(advantage).all():
        raise ValueError('Every method requires the same complete fixed forecast bank')
    flags = evidence.safety_flags_by_mask
    if (len(flags) != len(MASKS) or any(len(row) != 3 for row in flags)
            or any(not isinstance(value, (bool, np.bool_)) for row in flags for value in row)):
        raise ValueError('Every method requires all eight three-agent physical-mask checks')
    if not all(flags[0]):
        raise ValueError('The common baseline must pass the common physical oracle')
    bank = np.asarray(evidence.proposed_bank_command, dtype=float)
    if bank.shape != (3,) or not np.isfinite(bank).all():
        raise ValueError('The proposed bank-performance gate requires one finite joint command')
    for name in ('proposed_bank_admitted', 'proposed_bank_physical_admissible',
                 'proposed_bank_prediction_complete'):
        if not isinstance(getattr(evidence, name), (bool, np.bool_)):
            raise ValueError(name + ' must be Boolean')
    _sha256(evidence.proposed_bank_replay_sha256, 'Proposed bank-performance replay')
    if (isinstance(evidence.proposed_bank_candidate_count, (bool, np.bool_))
            or not isinstance(evidence.proposed_bank_candidate_count, Integral)
            or not 0 <= evidence.proposed_bank_candidate_count <= 28):
        raise ValueError('Invalid proposed bank-performance candidate count')
    bank_differs = bool(np.any(np.abs(bank[active] - baseline[active]) > 1e-9))
    if evidence.proposed_bank_admitted != bank_differs:
        raise ValueError('Proposed bank-performance adoption differs from its executed command')
    if (np.any(np.abs(bank[~active] - baseline[~active]) > 1e-12)
            or not evidence.proposed_bank_physical_admissible
            or (evidence.proposed_bank_admitted and not evidence.proposed_bank_prediction_complete)
            or (not evidence.proposed_bank_admitted and not np.array_equal(bank, baseline))):
        raise ValueError('Invalid proposed bank-performance fallback, completion or physical evidence')
    _validate_ledger(evidence.ledger, spec.budget)
    if (not evidence.ledger.timed_out and evidence.ledger.proposed_gate_forecast_calls < 1):
        raise ValueError('A completed proposed bank-performance replay must record forecast work')
    if (evidence.ledger.timed_out and (evidence.proposed_bank_admitted
            or not np.array_equal(bank, baseline)
            or evidence.proposed_bank_prediction_complete)):
        raise ValueError('Timed-out proposed bank-performance evidence must be a baseline fallback')
    return evidence


def _mamps_mask(flags):
    mask = (True, True, True)
    for _ in range(5):
        outcome = tuple(flags[_mask_code(mask)])
        if all(outcome):
            return mask
        if not any(mask):
            raise RuntimeError('Common baseline failed its physical oracle')
        following = tuple(mask[i] and outcome[i] for i in range(3))
        mask = (False, False, False) if following == mask else following
    raise RuntimeError('Monotone MAMPS adapter did not terminate')


class FairComparisonProtocol:
    def __init__(self, spec):
        if not isinstance(spec, ProtocolSpec):
            raise ValueError('A fully hash-bound pre-registered protocol is required')
        self.spec = spec
        self.protocol_sha256 = fingerprint(self.spec)

    def select(self, method, evidence):
        if method not in METHODS:
            raise ValueError('Unknown or non-preregistered comparator')
        validate_evidence(evidence, self.spec)
        evidence_sha256 = fingerprint(evidence)
        flags = evidence.safety_flags_by_mask
        all_learning = (True, True, True)
        baseline = (False, False, False)
        score = threshold = None
        command_override = physical_override = None
        if evidence.ledger.timed_out:
            mask, reason = baseline, 'common_evidence_timeout'
        elif method == 'proposed_bank_performance_gate':
            command_override = tuple(map(float, evidence.proposed_bank_command))
            mask = tuple(abs(command_override[i] - evidence.baseline[i]) > 1e-9 for i in range(3))
            physical_override = bool(evidence.proposed_bank_physical_admissible)
            reason = ('proposed_bank_performance_admitted' if evidence.proposed_bank_admitted
                      else 'proposed_bank_performance_fallback')
        elif method == 'always_learned':
            mask, reason = all_learning, 'negative_control_always_learned'
        elif method == 'always_baseline':
            mask, reason = baseline, 'negative_control_always_baseline'
        elif method == 'uncalibrated_confidence_threshold':
            score, threshold = float(evidence.confidence), self.spec.uncalibrated_confidence_threshold
            accepted = score >= threshold and all(flags[7])
            mask, reason = (all_learning, 'confidence_and_physical_pass') if accepted else (
                baseline, 'confidence_or_physical_reject')
        elif method == 'existing_calibrated_gate':
            score = float(evidence.predicted_delta_s + evidence.calibration_quantile_s)
            threshold = -self.spec.calibrated_minimum_gain_s
            accepted = (evidence.in_calibration_domain and evidence.prediction_complete
                        and score <= threshold and all(flags[7]))
            mask, reason = (all_learning, 'calibrated_upper_cost_pass') if accepted else (
                baseline, 'calibrated_domain_prediction_cost_or_physical_reject')
        elif method == 'advantage_lcb_adapter':
            samples = np.asarray(evidence.advantage_samples_s, dtype=float)
            score = float(samples.mean() - self.spec.advantage_lcb_multiplier
                          * samples.std(ddof=1) / np.sqrt(len(samples)))
            threshold = self.spec.advantage_min_gain_s
            accepted = evidence.prediction_complete and score > threshold and all(flags[7])
            mask, reason = (all_learning, 'advantage_lcb_and_physical_pass') if accepted else (
                baseline, 'advantage_lcb_prediction_or_physical_reject')
        elif method == 'mps_adapter':
            mask = all_learning if all(flags[7]) else baseline
            reason = 'mps_all_or_none_physical_switch'
        else:
            mask, reason = _mamps_mask(flags), 'mamps_monotone_physical_switch'
        command = command_override if command_override is not None else _command(evidence, mask)
        adopted = any(evidence.active[i] and abs(command[i] - evidence.baseline[i]) > 1e-9
                      for i in range(3))
        descriptor = DESCRIPTORS[method]
        return Decision(method=method, command=command, learning_mask=mask, adopted=adopted,
            physical_admissible=(physical_override if physical_override is not None
                                 else bool(all(flags[_mask_code(mask)]))), reason=reason,
            evidence_sha256=evidence_sha256, decision_rule_status=descriptor.decision_rule_status,
            closed_loop_status=descriptor.closed_loop_status, score=score, threshold=threshold)

    def select_all(self, evidence):
        decisions = tuple(self.select(method, evidence) for method in METHODS)
        if len({item.evidence_sha256 for item in decisions}) != 1:
            raise RuntimeError('Every comparator must consume the identical evidence bytes')
        return decisions


def fair_schedule(cases):
    cases = tuple(cases)
    if not cases or any(not isinstance(case, CaseIdentity) for case in cases):
        raise ValueError('At least one explicit fair-comparison case is required')
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError('Fair-comparison case IDs must be unique')
    assignments = []
    for index, case in enumerate(cases):
        offset = index % len(METHODS)
        ordered = METHODS[offset:] + METHODS[:offset]
        if index % 2:
            ordered = ordered[::-1]
        assignments.extend(Assignment(case, method, order) for order, method in enumerate(ordered))
    return tuple(assignments)


def _record_arrays(record):
    cost = np.asarray(record.per_vehicle_cost, dtype=float)
    arrival = np.asarray(record.per_vehicle_arrival, dtype=float)
    if (cost.shape != (3,) or arrival.shape != (3,) or not np.isfinite(cost).all()
            or not np.isfinite(arrival).all() or np.any(cost < 0.) or np.any(arrival < 0.)):
        raise ValueError('Finite capped three-vehicle costs and arrivals are required')
    if not np.isclose(float(record.team_cost), cost.mean(), atol=1e-10, rtol=0):
        raise ValueError('Team cost must equal the registered per-vehicle mean')
    return cost, arrival


def _exact_upper(k, n, error):
    if n == 0 or k == n:
        return 1.
    return float(beta.ppf(1. - error, k + 1, n - k))


def _exact_lower(k, n, error):
    if n == 0 or k == 0:
        return 0.
    return float(beta.ppf(error, k, n - k + 1))


def summarize_matched_records(records, spec):
    """Audit a complete paired grid and apply only the pre-registered ranking.

    Independent-cluster aggregation prevents repeated scene/dynamics cells from
    being counted as new statistical samples.  Gain lower bounds are asymptotic
    paired Gaussian diagnostics; conditional-harm and adoption bounds are exact.
    """
    if not isinstance(spec, ProtocolSpec):
        raise ValueError('A fully hash-bound pre-registered protocol is required')
    protocol_sha256 = fingerprint(spec)
    records = tuple(records)
    if not records:
        raise ValueError('A complete matched record grid is required')
    by_case, identities = {}, {}
    for record in records:
        if not isinstance(record, EpisodeRecord) or record.method not in METHODS:
            raise ValueError('Only pre-registered episode records are allowed')
        if (record.protocol_sha256 != protocol_sha256 or record.cost_contract != spec.cost_contract
                or not np.isclose(record.timeout_ms, spec.budget.decision_timeout_ms, atol=0., rtol=0.)):
            raise ValueError('Every method must share the frozen protocol, timeout and cost')
        if not isinstance(record.adopted, (bool, np.bool_)):
            raise ValueError('Adoption must be Boolean')
        if any(not isinstance(getattr(record, name), (bool, np.bool_))
               for name in ('completed', 'timed_out', 'collision', 'aborted')):
            raise ValueError('Completion and failure flags must be Boolean')
        if (isinstance(record.contract_violations, (bool, np.bool_))
                or not isinstance(record.contract_violations, Integral)
                or record.contract_violations < 0):
            raise ValueError('Contract violations must be a nonnegative integer')
        _record_arrays(record)
        previous = identities.setdefault(record.case.case_id, record.case)
        if previous != record.case:
            raise ValueError('One case ID cannot change scene, initial, dynamics or disturbance')
        methods = by_case.setdefault(record.case, {})
        if record.method in methods:
            raise ValueError('Repeated method/case record')
        methods[record.method] = record
    if any(set(methods) != set(METHODS) for methods in by_case.values()):
        raise ValueError('Every case must contain the complete comparator family exactly once')

    tolerance = 1e-9
    case_values = {}
    for case, methods in by_case.items():
        baseline = methods['always_baseline']
        if baseline.adopted:
            raise ValueError('Always-baseline cannot contain a learning adoption')
        baseline_cost, baseline_arrival = _record_arrays(baseline)
        baseline_failed = bool(not baseline.completed or baseline.timed_out
                               or baseline.collision or baseline.aborted
                               or baseline.contract_violations)
        values = {}
        for method in METHODS:
            record = methods[method]
            cost, arrival = _record_arrays(record)
            failed = bool(not record.completed or record.timed_out
                          or record.collision or record.aborted
                          or record.contract_violations)
            if method != 'always_baseline' and not record.adopted:
                if (not np.allclose(cost, baseline_cost, atol=tolerance, rtol=0)
                        or not np.allclose(arrival, baseline_arrival, atol=tolerance, rtol=0)
                        or failed != baseline_failed):
                    raise ValueError('A zero-adoption comparator must equal its paired baseline')
            harmful = bool(failed or np.any(cost > baseline_cost + tolerance)
                           or np.any(arrival > baseline_arrival + tolerance))
            values[method] = dict(adopted=bool(record.adopted), harmful=harmful,
                gain=float(baseline.team_cost - record.team_cost), baseline_cost=float(baseline.team_cost))
        case_values[case] = values

    clusters = {}
    for case, values in case_values.items():
        clusters.setdefault(case.independent_cluster, []).append(values)
    method_clusters = {method: [] for method in METHODS}
    for cluster_values in clusters.values():
        for method in METHODS:
            rows = [item[method] for item in cluster_values]
            method_clusters[method].append(dict(adopted=any(row['adopted'] for row in rows),
                harmful=any(row['harmful'] for row in rows),
                gain=float(np.mean([row['gain'] for row in rows])),
                baseline_cost=float(np.mean([row['baseline_cost'] for row in rows]))))

    risk_error = spec.family_error * spec.risk_error_share / len(CANDIDATE_METHODS)
    adoption_error = spec.family_error * spec.adoption_error_share / len(CANDIDATE_METHODS)
    gain_error = spec.family_error * spec.gain_error_share / len(CANDIDATE_METHODS)
    summaries = {}
    for method in METHODS:
        rows = method_clusters[method]
        adopted = np.asarray([row['adopted'] for row in rows], dtype=bool)
        harmful = np.asarray([row['harmful'] for row in rows], dtype=bool)
        gains = np.asarray([row['gain'] for row in rows], dtype=float)
        baseline_costs = np.asarray([row['baseline_cost'] for row in rows], dtype=float)
        n, a, h = len(rows), int(adopted.sum()), int((adopted & harmful).sum())
        if np.any(baseline_costs <= 0.):
            raise ValueError('Positive capped baseline cost is required for percentage gain')
        mean_gain, baseline_mean = float(gains.mean()), float(baseline_costs.mean())
        cluster_gain_percent = 100. * gains / baseline_costs
        mean_cluster_gain_percent = float(cluster_gain_percent.mean())
        ratio_of_means_gain_percent = 100. * mean_gain / baseline_mean
        percent_sd = float(cluster_gain_percent.std(ddof=1)) if n > 1 else 0.
        critical = float(t.ppf(1. - gain_error, n - 1)) if n > 1 else None
        gain_lower_percent = (None if n < 2 or percent_sd <= 1e-12 else
            float(mean_cluster_gain_percent - critical * percent_sd / np.sqrt(n)))
        adoption_lower = _exact_lower(a, n, adoption_error)
        risk_upper = _exact_upper(h, a, risk_error)
        eligible = bool(method != 'always_baseline' and adoption_lower >= spec.adoption_minimum
                        and gain_lower_percent is not None
                        and gain_lower_percent >= spec.gain_minimum_percent)
        summaries[method] = dict(independent_clusters=n, adopted_clusters=a,
            harmful_adopted_clusters=h, adoption_rate=float(a/n),
            simultaneous_adoption_lower=adoption_lower,
            empirical_conditional_harm=float(h/a) if a else None,
            simultaneous_conditional_harm_upper=risk_upper,
            mean_paired_team_gain_s=mean_gain,
            mean_paired_cluster_gain_percent=mean_cluster_gain_percent,
            ratio_of_means_gain_percent=ratio_of_means_gain_percent,
            paired_percent_standard_deviation=percent_sd,
            paired_percent_t_critical=critical,
            simultaneous_paired_t_gain_lower_percent=gain_lower_percent,
            usefulness_constraints_passed=eligible, descriptor=asdict(DESCRIPTORS[method]))
    eligible = [method for method in CANDIDATE_METHODS
                if summaries[method]['usefulness_constraints_passed']]
    winner = min(eligible, key=lambda method:(
        summaries[method]['simultaneous_conditional_harm_upper'],
        -summaries[method]['simultaneous_paired_t_gain_lower_percent'],
        -summaries[method]['simultaneous_adoption_lower'], METHODS.index(method))) if eligible else None
    return dict(protocol_sha256=protocol_sha256, primary_metric=spec.primary_metric,
        family_error=spec.family_error,
        error_allocation=dict(risk_family=spec.family_error*spec.risk_error_share,
            adoption_family=spec.family_error*spec.adoption_error_share,
            gain_family=spec.family_error*spec.gain_error_share,
            candidate_methods=len(CANDIDATE_METHODS), risk_per_method=risk_error,
            adoption_per_method=adoption_error, gain_per_method=gain_error),
        independent_clusters=len(clusters), cases=len(by_case), methods=summaries,
        adoption_constraint=spec.adoption_minimum, gain_constraint_percent=spec.gain_minimum_percent,
        always_baseline_eligible=False, selected_method=winner,
        exact_bounds=('conditional harm and adoption use exact one-sided binomial bounds; '
                      'gain uses a Bonferroni one-sided paired t bound on per-cluster percentage '
                      'gains, exact under IID normal percentage gains and otherwise diagnostic'),
        full_reproduction_and_adapters_explicit=True)
