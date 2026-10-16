import pytest
import torch
from coopgate.ppo_numerics_v15 import logspace_categorical_kl,require_complete_training


def test_finite_logit_underflow_is_not_a_missing_support_event():
    p=torch.tensor([[0.,-80.,-2.]])
    q=torch.tensor([[0.,-105.,-1.]],requires_grad=True)
    mask=torch.ones_like(p,dtype=torch.bool)
    reference=torch.distributions.Categorical(logits=p)
    current=torch.distributions.Categorical(logits=q)
    assert torch.isinf(torch.distributions.kl_divergence(reference,current)).all()
    stable=logspace_categorical_kl(p,q,mask)
    exact=torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=p.double()),torch.distributions.Categorical(logits=q.double()))
    torch.testing.assert_close(stable.double(),exact,atol=1e-7,rtol=1e-6)
    stable.sum().backward();assert torch.isfinite(q.grad).all()


def test_masked_logits_do_not_contribute_and_regular_kl_matches():
    p=torch.tensor([[1.,2.,3.],[3.,1.,2.]])
    q=torch.tensor([[2.,3.,1.],[2.,3.,1.]],requires_grad=True)
    mask=torch.tensor([[True,True,False],[True,False,True]])
    a=torch.distributions.Categorical(logits=p.masked_fill(~mask,-1e9))
    b=torch.distributions.Categorical(logits=q.masked_fill(~mask,-1e9))
    actual=logspace_categorical_kl(p,q,mask)
    torch.testing.assert_close(actual,torch.distributions.kl_divergence(a,b))
    actual.sum().backward();assert torch.all(q.grad[~mask]==0)


def test_failed_partial_training_cannot_be_used_as_final_model():
    meta=dict(steps=114688,target_steps=262144,actor_parameter_delta_l2=[1.,2.,3.])
    with pytest.raises(ValueError,match='fixed final'):require_complete_training(meta)
    meta['steps']=262144;require_complete_training(meta)
    meta['actor_parameter_delta_l2'][1]=float('nan')
    with pytest.raises(ValueError,match='three actors'):require_complete_training(meta)
