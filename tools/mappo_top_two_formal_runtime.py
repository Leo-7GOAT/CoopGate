"""Attempt-6 execution adapters; unchanged physics and complete online timing.

The nominal/thinned comparator receives the SAME bounded rank-two candidate
provider as the target. MPS/MAMPS remain disclosed raw-proposal adapters.
"""
from types import FunctionType
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mappo_powered_history_formal_runtime as old
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate
from coopgate.episode_thinning_control import EpisodeThinnedNominal
from experiments.top_two_candidates.gate import TopTwoHistoryGate, TopTwoMixin
from experiments.top_two_candidates.audit import audit_record as ranked_audit

MatchedShiftedMergeEnv = old.MatchedShiftedMergeEnv
TARGET_METHODS = old.TARGET_METHODS
CALIBRATION_METHODS = old.CALIBRATION_METHODS
CONFIRMATION_METHODS = old.CONFIRMATION_METHODS
METHODS = old.METHODS
COUPLED_BANK = old.COUPLED_BANK
check_calls = old.check_calls
truthful_rule = old.truthful_rule


class ScalarNominalOnlyGate(old.ScalarNominalOnlyGate):
    _candidate_union = TopTwoMixin._candidate_union

    def __init__(self, model, profile):
        super().__init__(model, profile)
        original = RecoveryJointGate.select
        scope = dict(original.__globals__)
        scope['compositions'] = self._candidate_union
        self._ranked_select = FunctionType(original.__code__, scope,
            name='same_pool_nominal_selector', argdefs=original.__defaults__)

    def select(self, state, previous, unused=None):
        self._top_two_evidence = None
        command, info = self._ranked_select(self, state, previous, unused)
        info.update(candidate_proposal_source='local_masked_top_two',
                    top_two_candidates=self._top_two_evidence,
                    maximum_extra_candidates=4,
                    logits_are_feasibility_probabilities=False,
                    deployed_critic_used=False,
                    nominal_control_uses_same_ranked_candidate_provider=True)
        return command, info


def nominal_factory(model, profile, rule):
    if rule == 'baseline':
        return old.MemoizedRecoveryGate(None, profile, 'baseline')
    if rule == 'nominal_performance':
        return ScalarNominalOnlyGate(model, profile)
    raise ValueError('Only the identical baseline or same-pool nominal branch')


def make_controller(method, model, case):
    profile = case['plant']
    truthful_rule(method, case)
    if method in TARGET_METHODS:
        return TopTwoHistoryGate(model, profile, COUPLED_BANK)
    if method.startswith('thinned_'):
        seed = method.split('_')[1]
        return EpisodeThinnedNominal(model, profile, case['controller_coins'][seed], factory=nominal_factory)
    if method.startswith('nominal_'):
        return ScalarNominalOnlyGate(model, profile)
    return old.make_controller(method, model, case)


def execute_episode(case, method, model, env, *, controller_factory=make_controller):
    return old.execute_episode(case, method, model, env, controller_factory=controller_factory)


def audit_record(case, record, model, *, reference_factory=make_controller):
    return ranked_audit(case, record, model, reference_factory=reference_factory)
