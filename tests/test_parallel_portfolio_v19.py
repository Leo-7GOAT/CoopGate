from concurrent.futures import ThreadPoolExecutor

import numpy as np

from coopgate.bounded_portfolio_v19 import bounded_action_then_baseline
from coopgate.cached_portfolio_v19 import cached_disturbed_suffix, cached_nominal_suffix, new_command_cache
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.parallel_portfolio_v19 import evaluate_disturbed_candidate
from coopgate.stress_prediction_v17 import fixed_stress_bank


def test_concurrent_complete_forecasts_keep_independent_mutable_caches():
    _, tapes = fixed_stress_bank()
    packed = parameters(MergeConfig())
    state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    previous = np.zeros(3)
    source = new_command_cache()
    _, command, _, _ = cached_nominal_suffix(state, previous, 0, packed, source)
    entries = dict(source)
    private = [source.copy(), source.copy()]
    # Compile before concurrent invocation; the benchmark also warms kernels.
    evaluate_disturbed_candidate(state, previous, 0, command, tapes[1], packed,
                                  source.copy(), False, np.zeros((3, 2)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate_disturbed_candidate, state, previous, 0, command,
                               tapes[index], packed, cache, False, np.zeros((3, 2)))
                   for index, cache in zip((1, 9), private)]
        results = [future.result() for future in futures]
    assert dict(source) == entries
    for index, result in zip((1, 9), results):
        cache = new_command_cache()
        baseline, _, _ = cached_disturbed_suffix(state, previous, 0, tapes[index], packed, cache)
        value, _, _, pruned, _ = bounded_action_then_baseline(
            state, previous, 0, command, tapes[index], packed, baseline, cache, False)
        np.testing.assert_array_equal(result[0], baseline)
        np.testing.assert_array_equal(result[1], value - baseline)
        assert result[4] == pruned
    assert private[0] is not private[1]
