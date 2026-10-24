"""Real evidence generation for the fixed eight-selector comparison.

Evidence is shared by exact observed state AND executed command history within
one paired case. Cached packets retain their original measured deadline outcome;
their attributed cost is not counted as another actual computation. This is a
comparison of selectors on a shared evidence graph, not standalone latency.
"""
from copy import deepcopy
from dataclasses import asdict
import time

import numpy as np

from .actuator_recovery_v34 import memory_step
from .confidence_v12 import conflict_group, in_training_domain
from .fair_comparison_v39 import (MASKS, DecisionEvidence, FairComparisonProtocol,
                                 WorkLedger, fingerprint, validate_evidence)
from .recovery_joint_gate_v35 import RecoveryJointGate, actuator_certificate_flags


SCORE_SCALE_SECONDS = .05


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k != 'gate_ms'}
    if isinstance(value, (tuple, list)):
        return [without_timing(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class EvidenceDeadline(RuntimeError):
    pass


class EvidenceFailure(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(record['error']['message'])


class EvidenceFactory:
    """One case, one model, no access to randomized plant or future tape."""
    def __init__(self, case, spec, model, calibration):
        self.case, self.spec, self.model = case, spec, model
        self.calibration = deepcopy(calibration)
        self.cache = {}
        self.actual_builds = self.cache_hits = 0

    def get(self, state, history):
        identity = dict(case=asdict(self.case), observed=np.asarray(state).tolist(),
                        executed_commands=np.asarray(history, dtype=float).reshape(-1, 3).tolist())
        key = fingerprint(identity)
        hit = key in self.cache
        if hit:
            self.cache_hits += 1
        else:
            self.actual_builds += 1
            self.cache[key] = self.build(state, history)
        evidence, record = deepcopy(self.cache[key])
        if record['error'] is not None:
            raise EvidenceFailure(record)
        return evidence, record, hit, key

    def build(self, state, history, *, replay_clock=None):
        start = time.perf_counter()
        ticks, replay_index = [], 0
        x = np.asarray(state, dtype=float).copy()
        commands = np.asarray(history, dtype=float).reshape(-1, 3)
        step = len(commands)
        previous = commands[-1].copy() if step else np.zeros(3)
        if x.shape != (3, 2) or not np.isfinite(x).all() or step >= 50:
            raise ValueError('Finite observed state and an unexhausted command history required')
        if not np.isfinite(commands).all():
            raise ValueError('Finite executed command history required')
        work = dict(observation_reads=1, learned_proposal_calls=0, baseline_proposal_calls=0,
                    forecast_paths=0, safety_mask_checks=0, proposed_gate_replays=1,
                    proposed_gate_forecast_calls=0)
        details = dict(baseline=None, proposal=None, bank=None, raw_paths=[], raw_kernel_calls=0)

        def clock():
            nonlocal replay_index
            if replay_clock is None:
                value = 1000. * (time.perf_counter() - start)
            else:
                if replay_index >= len(replay_clock):
                    raise ValueError('Recorded deadline checkpoints exhausted')
                value = float(replay_clock[replay_index])
                replay_index += 1
            if not np.isfinite(value) or value < 0 or (ticks and value < ticks[-1]):
                raise ValueError('Invalid monotone deadline trace')
            ticks.append(value)
            return value

        def check():
            if clock() >= self.spec.budget.decision_timeout_ms:
                raise EvidenceDeadline('Common evidence deadline reached')

        factory = self

        class CountedModel:
            def proposal(self, observations, mask):
                check()
                result = factory.model.proposal(observations, mask)
                work['learned_proposal_calls'] += 1
                details['proposal'] = np.asarray(result).tolist()
                return result

        gate = RecoveryJointGate(CountedModel(), self.case.dynamics, 'bank_performance')
        # select advances memory using the latest previous command. Restore only
        # commands before that one, so each command contributes exactly once.
        for command in commands[:-1]:
            for _ in range(5):
                gate.memory = gate.retention * gate.memory + (1 - gate.retention) * command
                gate.baseline.memory_lower, gate.baseline.memory_upper = memory_step(
                    gate.baseline.memory_lower, gate.baseline.memory_upper, command, gate.bounds)
        gate.step = gate.baseline.step = step
        original_baseline, original_predict = gate.baseline.select, gate.predict

        def baseline_call(*args, **kwargs):
            result = original_baseline(*args, **kwargs)
            work['baseline_proposal_calls'] += 1
            details['baseline'] = dict(command=np.asarray(result[0]).tolist(),
                                       info=without_timing(result[1]))
            check()  # retain a valid baseline even when its computation is late
            return result

        def bank_predict(*args, **kwargs):
            check()
            result = original_predict(*args, **kwargs)
            work['proposed_gate_forecast_calls'] += 1
            return result

        gate.baseline.select, gate.predict = baseline_call, bank_predict
        flags = [[True] * 3] + [[False] * 3 for _ in range(7)]
        advantages = [0.] * 33
        complete, timed_out, error = True, False, None
        chosen, info = None, None
        try:
            chosen, info = gate.select(x, previous)
            details['bank'] = dict(command=chosen.tolist(), info=without_timing(info))
            check()
            base, proposal = np.array(info['baseline_command']), np.array(info['actor_proposal'])
            mlo, mhi = np.array(info['actuator_memory_lower']), np.array(info['actuator_memory_upper'])
            for code, mask in enumerate(MASKS):
                check()
                candidate = np.where(mask, proposal, base)
                flags[code] = actuator_certificate_flags(x, previous, candidate, mlo, mhi,
                                                          gate.bounds, gate.packed).tolist()
                work['safety_mask_checks'] += 1
            gate.step = step  # raw candidate forecasts use the current decision clock
            for path in range(33):
                check()
                before = original_predict(x, previous, mlo, mhi, None, path)
                details['raw_kernel_calls'] += 1
                check()
                after = original_predict(x, previous, mlo, mhi, proposal, path)
                details['raw_kernel_calls'] += 1
                valid = bool(before[2] and after[2])
                gain = float(np.mean(before[0][:, 0] - after[0][:, 0])) if valid else 0.
                advantages[path] = gain
                complete &= valid
                details['raw_paths'].append(dict(path=path, complete=valid, advantage_s=gain,
                    baseline_values=before[0].tolist(), candidate_values=after[0].tolist(),
                    baseline_error=int(before[3]), candidate_error=int(after[3])))
                work['forecast_paths'] += 1
            check()
        except EvidenceDeadline:
            timed_out, complete = True, False
        except (RuntimeError, ValueError, IndexError, TypeError) as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
        record = dict(case=asdict(self.case), observed=x.tolist(), history=commands.tolist(),
                      checkpoints_ms=ticks, work=work, details=details, error=error)
        if replay_clock is not None and replay_index != len(replay_clock):
            raise ValueError('Unused recorded deadline checkpoints')
        if error is not None:
            return None, record
        if details['baseline'] is None:
            raise RuntimeError('No baseline exists for common timeout fallback')
        base = np.array(details['baseline']['command'])
        active = x[:, 0] < gate.c.goal
        proposal = np.array(info['actor_proposal']) if info is not None else base.copy()
        if timed_out:
            chosen = base.copy()
        if chosen is None:
            raise RuntimeError('Missing bank output without a timeout')
        raw_nominal_complete = bool(details['raw_paths'] and details['raw_paths'][0]['complete'])
        predicted_delta = -advantages[0]
        confidence = float(1 / (1 + np.exp(np.clip(predicted_delta / SCORE_SCALE_SECONDS, -50, 50))))
        if not raw_nominal_complete:
            confidence = 0.
        group = conflict_group(x)
        quantile = float(self.calibration['groups'][group]['q'])
        evidence = DecisionEvidence(case=self.case, step=step,
            observation_fingerprint=fingerprint(dict(state=x.tolist(), previous=previous.tolist(),
                step=step, profile=self.case.dynamics, history=commands.tolist())),
            proposal=tuple(proposal), baseline=tuple(base), active=tuple(active),
            confidence=confidence, predicted_delta_s=predicted_delta, calibration_quantile_s=quantile,
            in_calibration_domain=bool(in_training_domain(x)), prediction_complete=bool(complete),
            advantage_samples_s=tuple(advantages), safety_flags_by_mask=tuple(tuple(row) for row in flags),
            proposed_bank_command=tuple(chosen),
            proposed_bank_admitted=bool(np.any(np.abs(chosen[active] - base[active]) > 1e-9)),
            proposed_bank_physical_admissible=True,
            proposed_bank_prediction_complete=bool(not timed_out and info is not None
                and info['reference_baseline_complete']
                and (not info['actual_learning_used'] or len(info['selected_bank_differences']) == 33)),
            proposed_bank_replay_sha256=fingerprint(details),
            proposed_bank_candidate_count=int(info['candidate_count']) if info is not None else 0,
            ledger=WorkLedger(**work, elapsed_ms=ticks[-1], timed_out=timed_out))
        validate_evidence(evidence, self.spec)
        record['evidence_sha256'] = fingerprint(evidence)
        return evidence, record

    def audit(self, evidence, record):
        """Recompute controller/forecast evidence with recorded clock checkpoints.

        Replay validates deterministic outputs and the measured cutoff decisions;
        replay wall time cannot validate the earlier operating-system duration.
        """
        expected, proof = self.build(record['observed'], record['history'],
                                     replay_clock=record['checkpoints_ms'])
        if fingerprint(expected) != fingerprint(evidence) or fingerprint(proof) != fingerprint(record):
            raise ValueError('Independent evidence replay differs')
        return dict(replayed=True, timeout=bool(evidence and evidence.ledger.timed_out),
                    earlier_wall_time_independently_certified=False)


class FairSelectorGate:
    def __init__(self, method, factory):
        self.method, self.factory = method, factory
        self.protocol = FairComparisonProtocol(factory.spec)
        self.history, self.calls = [], []

    def select(self, state, previous, unused=None):
        previous = np.asarray(previous)
        expected = np.array(self.history[-1]) if self.history else np.zeros(3)
        np.testing.assert_array_equal(previous, expected)
        try:
            evidence, proof, hit, key = self.factory.get(state, self.history)
        except EvidenceFailure as exc:
            self.calls.append(dict(error=exc.record))
            raise
        decision = self.protocol.select(self.method, evidence)
        command = np.array(decision.command)
        self.calls.append(dict(error=None, packet_key=key, cache_hit=hit,
                               decision=asdict(decision), evidence=asdict(evidence), proof=proof))
        self.history.append(command.tolist())
        return command, dict(in_set=decision.physical_admissible, certified=decision.physical_admissible,
            intervention=float(not decision.adopted), actual_learning_used=decision.adopted,
            baseline_command=list(evidence.baseline), evidence_key=key,
            evidence_sha256=decision.evidence_sha256, common_evidence_timeout=evidence.ledger.timed_out,
            attributed_evidence_ms=evidence.ledger.elapsed_ms, evidence_cache_hit=hit,
            decision_reason=decision.reason, rule=self.method)
