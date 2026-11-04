"""KL on finite action logits without treating float underflow as zero support."""
import torch


def logspace_categorical_kl(reference_logits, current_logits, mask):
    if reference_logits.shape != current_logits.shape or mask.shape != current_logits.shape:
        raise ValueError('Matching categorical shapes required')
    if not mask.any(-1).all() or not torch.isfinite(reference_logits).all() or not torch.isfinite(current_logits).all():
        raise ValueError('Finite logits and nonempty support required')
    logp = torch.log_softmax(reference_logits.masked_fill(~mask, -1e9), dim=-1)
    logq = torch.log_softmax(current_logits.masked_fill(~mask, -1e9), dim=-1)
    return (logp.exp()*(logp-logq)).masked_fill(~mask, 0.).sum(-1)


def require_complete_training(meta):
    if meta.get('steps', 0) <= 0 or meta.get('steps') != meta.get('target_steps'):
        raise ValueError('Only the fixed final training budget is eligible for comparison')
    changes = meta.get('actor_parameter_delta_l2', [])
    if len(changes) != 3 or not all(torch.isfinite(torch.tensor(v)) and v > 1e-8 for v in changes):
        raise ValueError('All three actors must actually have trained')
