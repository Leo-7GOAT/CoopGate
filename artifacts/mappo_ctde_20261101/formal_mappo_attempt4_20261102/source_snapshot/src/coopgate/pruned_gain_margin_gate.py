# naive _select sha256=b6c0d003e16c0a34c54becc46f42038a8a4c5e3bb3cf8c4233e8544be76bd284
"""Reject candidates as soon as a computed path upper-bounds min confidence below floor.

All admitted candidates retain complete original evidence;no extra admission,
new actor,altered nominal ranking or weakened physical certificate.
"""
import time
import numpy as np
from .gain_margin_gate import ScalarGainMarginGate,relative_gain_score
from .actuator_recovery_v34 import validate_inputs,stopping_certificate
from .ego_v13 import ego_observations
from .marl_v12 import LEVELS,action_mask
from .physical_switch_v27 import recovery_search
from .recovery_joint_gate_v35 import actuator_certificate_flags,compositions
from .temporal_portfolio_v20 import BANK_STREAM

class PrunedGainMarginGate(ScalarGainMarginGate):

    def _select(self, state, previous, unused=None):
        self.gain_margin_prefix_rejections = []
        self.gain_margin_evidence = []
        self.selected_gain_margin = None
        self.selected_baseline_team_costs = []
        started = time.perf_counter()
        x, prev, _, _, _ = validate_inputs(state, previous, self.baseline.memory_lower, self.baseline.memory_upper, self.bounds, self.c)
        if self.baseline.step != self.step:
            raise RuntimeError('Baseline and gate clocks differ')
        if self.step:
            for _ in range(round(self.c.decision_dt / self.c.dt)):
                self.memory = self.retention * self.memory + (1.0 - self.retention) * prev
        base, base_info = self.baseline.select(x, prev)
        mlo, mhi = (np.array(base_info['actuator_memory_lower']), np.array(base_info['actuator_memory_upper']))
        if np.any(self.memory < mlo - 1e-12) or np.any(self.memory > mhi + 1e-12):
            raise RuntimeError('Reference actuator point escaped the command-history interval')
        proposal = base.copy()
        indices = None
        active = x[:, 0] < self.c.goal
        if self.rule != 'baseline':
            mask = action_mask(prev, self.c)
            indices = np.asarray(self.model.proposal(ego_observations(x, prev, self.step, self.c), mask))
            if indices.shape != (3,) or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= len(LEVELS)) or (not mask[np.arange(3), indices].all()):
                raise ValueError('Learned proposal violates the common action mask')
            proposal = LEVELS[indices].copy()
            proposal[~active] = base[~active]
        chosen = base.copy()
        candidate_evidence, nominal_evidence, masks, flags, winner, phases = ([], [], [], [], [], [])
        candidate_count = original_count = calls = transitions = planning = checks = 0
        planning_complete = True
        selected_code, selected_weight, margin, baseline_complete = (0, None, None, None)
        if self.rule in ('mps_shared', 'mamps_shared'):
            checker = lambda command: actuator_certificate_flags(x, prev, command, mlo, mhi, self.bounds, self.packed)
            chosen, _, masks, flags = recovery_search(proposal, base, checker, self.rule)
        elif self.rule in ('nominal_performance', 'bank_performance'):
            candidates, original_count = compositions(x, prev, proposal, base, mlo, mhi, self.bounds, self.packed)
            candidate_count = len(candidates)
            baseline_values, candidate_values = ({}, {})

            def predict(command, path):
                nonlocal calls, transitions, planning, checks, planning_complete
                value = self.predict(x, prev, mlo, mhi, command, path)
                calls += 1
                transitions += int(value[4])
                planning += int(value[5])
                checks += int(value[6])
                planning_complete = planning_complete and bool(value[21])
                return value
            baseline_values[0] = predict(None, 0)
            baseline_complete = bool(baseline_values[0][2])
            ranked = []
            for index, (code, weight, command, _) in enumerate(candidates):
                after = predict(command, 0)
                candidate_values[index] = after
                if not after[2] or not baseline_complete:
                    nominal_evidence.append(dict(candidate=index, baseline_complete=baseline_complete, candidate_complete=bool(after[2]), passed=False, differences=None, phase_differences=None))
                    continue
                delta = after[0] - baseline_values[0][0]
                phase_delta = after[1] - baseline_values[0][1]
                admissible = bool(np.all(delta <= 1e-10) and delta[:, 0].mean() < -1e-09 and np.all(phase_delta <= 0.05 + 1e-10))
                nominal_evidence.append(dict(candidate=index, baseline_complete=True, candidate_complete=True, passed=admissible, differences=delta.tolist(), phase_differences=phase_delta.tolist()))
                if admissible:
                    ranked.append((float(delta[:, 0].mean()), -weight, code, index))
            ranked.sort()
            path_count = 1 if self.rule == 'nominal_performance' else len(self.tapes)
            for _, _, code, index in ranked:
                candidate = candidates[index][2]
                differences, phase_differences = ([], [])
                failure = None
                passed = True
                for path in range(path_count):
                    if path == 0:
                        violating_paths = 0
                    if path not in baseline_values:
                        baseline_values[path] = predict(None, path)
                    before = baseline_values[path]
                    after = candidate_values[index] if path == 0 else predict(candidate, path)
                    if not before[2] or not after[2]:
                        passed = False
                        failure = dict(path=path, baseline_complete=bool(before[2]), candidate_complete=bool(after[2]), baseline_error=int(before[3]), candidate_error=int(after[3]))
                        break
                    delta = after[0] - before[0]
                    phase_delta = after[1] - before[1]
                    differences.append(delta.tolist())
                    phase_differences.append(phase_delta.tolist())
                    if np.any(delta > 1e-10) or np.any(phase_delta > 0.05 + 1e-10):
                        violating_paths += 1
                        if violating_paths > self.max_failed_paths:
                            passed = False
                            break
                    if self.gain_floor > 0:
                        denominator = float(before[0][:, 0].mean())
                        prefix_score = float(-delta[:, 0].mean() / denominator) if np.isfinite(denominator) and denominator > 0 else None
                        if prefix_score is None or not np.isfinite(prefix_score) or prefix_score < self.gain_floor:
                            self.gain_margin_prefix_rejections.append(dict(candidate=index, path=path, baseline_team_cost=denominator, score_upper_bound=prefix_score, floor=self.gain_floor))
                            passed = False
                            break
                candidate_evidence.append(dict(candidate=index, composition=code, passed=passed, failure=failure, checked_paths=len(differences), differences=differences, phase_differences=phase_differences, violating_paths=violating_paths))
                if passed:
                    if len(differences) != path_count:
                        raise RuntimeError('Incomplete performance evidence')
                    baseline_costs = [float(baseline_values[p][0][:, 0].mean()) for p in range(path_count)]
                    score = relative_gain_score(differences, baseline_costs)
                    floor_passed = self.gain_floor == 0 or (score is not None and score >= self.gain_floor)
                    self.gain_margin_evidence.append(dict(candidate=index, composition=code, repair_weight=candidates[index][1], baseline_team_costs=baseline_costs, worst_relative_team_gain=score, floor=self.gain_floor, passed=floor_passed))
                    if not floor_passed:
                        candidate_evidence[-1].update(passed=False, gain_floor_rejected=True)
                        continue
                    self.selected_gain_margin = score
                    self.selected_baseline_team_costs = baseline_costs
                    chosen, selected_code, selected_weight = (candidate.copy(), code, candidates[index][1])
                    winner, phases = (differences, phase_differences)
                    margin = min(-float(np.max(differences)), 0.05 - float(np.max(phase_differences)))
                    break
        physical = stopping_certificate(x, prev, chosen, mlo, mhi, self.bounds, self.packed)
        if not physical[0]:
            raise RuntimeError('Selected joint command lacks the common stopping certificate')
        if not np.array_equal(chosen[~active], base[~active]):
            raise RuntimeError('Cleared vehicles must retain their common baseline commands')
        used = bool(np.any(np.abs(chosen[active] - base[active]) > 1e-09))
        self.learning_used_ever |= used
        self.step += 1
        return (chosen, dict(in_set=True, certified=True, intervention=float(not used), gate_ms=1000 * (time.perf_counter() - started), rule=self.rule, actuator_class=self.profile, certificate_kind='bounded_actuator_memory_stopping', actuator_bounds=self.bounds.tolist(), actuator_memory_lower=mlo.tolist(), actuator_memory_upper=mhi.tolist(), baseline_command=base.tolist(), baseline_recovery_proposed=base_info['recovery_applied'], baseline_evidence={k: v for k, v in base_info.items() if k != 'gate_ms'}, recovery_applied=bool(not used and base_info['recovery_applied']), agent_actions=None if indices is None else indices.tolist(), actor_proposal=proposal.tolist(), actual_learning_used=used, actual_learning_used_ever=self.learning_used_ever, candidate_count=candidate_count, original_candidate_count=original_count, composition=selected_code, repair_weight=selected_weight, candidate_evidence=candidate_evidence, nominal_evidence=nominal_evidence, selected_bank_differences=winner, selected_phase_differences=phases, mode_search_masks=masks, mode_search_flags=flags, forecast_calls=calls, performance_forecast_transitions=transitions, completed_planning_model_transitions=planning, planning_work_complete=planning_complete, recovery_certificate_substeps=checks, prediction_parameters=self.reference, reconstructed_actuator=self.memory.tolist(), forecast_bank_stream=BANK_STREAM, forecast_bank_size=len(self.tapes), external_bank_radius=0.1, phase_tolerance_seconds=0.05, finite_bank_performance_margin=margin, margin_is_probability=False, reference_baseline_complete=baseline_complete, future_realization_used=False, true_randomized_plant_used=False, conditional_performance_risk_certified=False, scenario_pass_fraction=(path_count - violating_paths) / path_count if used else None, performance_confidence_kind='normalized_worst_path_team_gain', gain_confidence_floor=self.gain_floor, selected_gain_confidence_score=self.selected_gain_margin, selected_baseline_forecast_team_costs=self.selected_baseline_team_costs, gain_margin_evidence=self.gain_margin_evidence, gain_score_is_probability=False, gain_margin_prefix_rejections=self.gain_margin_prefix_rejections, confidence_pruning_kind='rejection_only_min_over_paths_upper_bound'))
