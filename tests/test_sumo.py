import numpy as np
import json
import pytest

from coopgate.sim import SumoEnv


def test_real_sumo_repeatable_reset_and_step():
    with SumoEnv(method='traditional') as env:
        first=env.reset(123)
        for _ in range(3):
            obs,r,term,trunc,info=env.step([4,4,4])
        final=obs.copy()
        again=env.reset(123)
        np.testing.assert_allclose(first,again,atol=1e-6)
        for _ in range(3):
            obs,*_=env.step([4,4,4])
        np.testing.assert_allclose(final,obs,atol=1e-6)
        assert np.isfinite(obs).all()
        for ident in env.ids[:3]:
            assert env.connection.vehicle.getSpeedMode(ident)==96
            assert env.connection.vehicle.getLaneChangeMode(ident)==0


@pytest.mark.parametrize('method',['mappo','independent','mamps_style','joint_nominal','joint_robust'])
def test_all_methods_produce_serializable_real_simulator_results(method):
    with SumoEnv(method=method) as env:
        env.reset(42)
        for _ in range(2):
            env.step([8,4,4])
        assert json.loads(json.dumps(env.trace))
        assert json.loads(json.dumps(env.metrics()))['method']==method
