"""Audited joint MAPPO actors wired to the existing actuator recovery gates.

New weights never inherit an older controller's risk certificate. Physical
class assumptions and the actual full gate require separate validation.
"""
import hashlib
import json
from pathlib import Path

from .mappo_ctde import load_mappo
from .memoized_recovery_v37 import MemoizedRecoveryGate
from .recovery_joint_gate_v35 import RULES


def load_deployment_actor(checkpoint):
    checkpoint=Path(checkpoint)
    audit_path=checkpoint.parent/'audit.json'
    if not audit_path.is_file():
        raise ValueError('Completed joint PPO rollout/gradient audit required')
    audit=json.loads(audit_path.read_text())
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if audit.get('policy_sha256')!=digest:
        raise ValueError('Audited MAPPO checkpoint changed')
    model,meta=load_mappo(checkpoint)
    if (meta['algorithm']!='mappo_ctde_20261101' or meta['critic_kind']!='central'
            or meta['stage']!='joint' or not 0<meta['steps']==meta['target_steps']
            or not model.config.role_actors):
        raise ValueError('Completed centralized-critic, three-independent-actor joint MAPPO required')
    deltas=audit.get('role_parameter_delta_squared_l2')
    if (not audit.get('on_policy_proposals_verified') or not audit.get('independent_gae_verified')
            or not audit.get('final_budget_complete') or deltas is None or len(deltas)!=3
            or not all(x>0 for x in deltas) or not all(n>0 for n in meta['actor_samples'])):
        raise ValueError('All three roles must have actual on-policy updates')
    return model,dict(policy_sha256=digest,audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        algorithm=meta['algorithm'],joint_steps=meta['steps'],seed=meta['seed'],
        ancestry=meta.get('inherited_legacy_initializer'),
        inherited_risk_certificate=False,learned_coordination_superiority_proved_by_loader=False)


def make_recovery_gate(model,profile,rule='bank_performance'):
    if rule not in RULES:
        raise ValueError('Use an explicit existing recovery/admission rule')
    if rule!='baseline' and model is None:
        raise ValueError('A loaded audited joint actor is required for learning rules')
    # Existing exact proposal-cache implementation; no shared offline evidence
    # wrapper is charged to the baseline or to another selector's timing.
    return MemoizedRecoveryGate(model,profile,rule)
