import numpy as np
import pytest
from coopgate.retention_set_observer import RetentionSetObserver


def test_true_parameters_and_memory_are_never_eliminated_with_bounded_innovations_and_clipping():
    rng=np.random.default_rng(640211)
    for profile in ('lag_mild','lag_strong'):
        for sample in range(32):
            observer=RetentionSetObserver(profile)
            rho=rng.uniform(observer.lower[:,0],observer.upper[:,-1]);memory=np.zeros(3)
            state=np.column_stack((rng.uniform(-200,-70,3),rng.choice([0.,1.,10.,19.9,20.],3)))
            observer.observe(state,np.zeros(3))
            true_cells=(observer.lower<=rho[:,None])&(rho[:,None]<=observer.upper)
            for _ in range(30):
                command=rng.uniform(-5,2.5,3)
                for sub in range(5):
                    memory=rho*memory+(1.-rho)*command
                    innovation=(np.full(3,.1 if sample%2 else -.1) if sample<2 else rng.uniform(-.1,.1,3))
                    state[:,1]=np.clip(state[:,1]+.1*(memory+innovation),0.,20.)
                    state[:,0]+=.1*state[:,1]
                status=observer.observe(state,command)
                assert status['consistent'] and observer.feasible[true_cells].all()
                assert np.all(np.broadcast_to(memory[:,None],observer.lower.shape)[true_cells]>=observer.memory_lower[true_cells]-1e-12)
                assert np.all(np.broadcast_to(memory[:,None],observer.lower.shape)[true_cells]<=observer.memory_upper[true_cells]+1e-12)
                lo,hi=observer.retention_bounds();assert np.all(rho>=lo) and np.all(rho<=hi)


def test_deterministic_history_reduces_uncertainty_and_uses_executed_command():
    observer=RetentionSetObserver('lag_strong');rho=np.exp(-.1/np.array([.31,.39,.48]));memory=np.zeros(3)
    state=np.array([[-100.,5.],[-140.,6.],[-180.,7.]])
    observer.observe(state,np.zeros(3))
    for k in range(8):
        command=np.full(3,2.5 if k%2==0 else -2.5)
        for _ in range(5):
            memory=rho*memory+(1.-rho)*command
            state[:,1]=np.clip(state[:,1]+.1*memory,0,20);state[:,0]+=.1*state[:,1]
        observer.observe(state,command)
    assert np.all(observer.feasible.sum(axis=1)<32)
    assert len(observer.commands)==8 and not observer.status()['parameter_posterior_probability']


def test_inconsistent_measurement_never_silently_resets_or_becomes_confident():
    observer=RetentionSetObserver('lag_strong');state=np.array([[-100.,5.],[-140.,6.],[-180.,7.]])
    observer.observe(state,np.zeros(3));state[:,1]=20.
    assert not observer.observe(state,np.zeros(3))['consistent']
    with pytest.raises(ValueError,match='No declared'):observer.retention_bounds()


def test_unsupported_models_and_nonzero_initial_memory_are_rejected():
    with pytest.raises(ValueError):RetentionSetObserver('gain_drag')
    with pytest.raises(ValueError):RetentionSetObserver('nominal')
    with pytest.raises(ValueError):RetentionSetObserver('lag_strong',64)
    with pytest.raises(ValueError):RetentionSetObserver('lag_strong').observe(np.zeros((3,2)),np.ones(3))
