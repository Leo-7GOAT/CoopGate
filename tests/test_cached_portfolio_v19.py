import numpy as np

from coopgate.cached_portfolio_v19 import cached_command, new_command_cache
from coopgate.deduplicated_planner_v19 import deduplicated_joint_rollout
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import joint_rollout
from coopgate.merge_v4 import MergeConfig


def test_deduplication_preserves_independent_reference_planner_on_different_phases():
    c = parameters(MergeConfig())
    for x, prev, step in (
        ([[-80., 12.], [-85., 13.], [-120., 11.]], [0., 0., 0.], 0),
        ([[-50., 6.], [-60., 4.], [-75., 6.]], [1.25, 0., 1.25], 5),
        ([[65., 20.], [45., 15.], [25., 15.]], [2.5, 1.25, 0.], 20),
        ([[100., 20.], [80., 20.], [65., 18.]], [2.5, 2.5, 1.25], 30),
    ):
        state, previous = np.array(x), np.array(prev)
        original = joint_rollout(state, previous, step, c)
        optimized = deduplicated_joint_rollout(state, previous, step, c)
        np.testing.assert_array_equal(original[0], optimized[0])
        np.testing.assert_array_equal(original[1:3], optimized[1:3])
        assert optimized[3] <= original[3]


def test_command_cache_does_not_quantize_state_or_drop_clock_and_memory():
    x = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    prev = np.zeros(3)
    packed = parameters(MergeConfig())
    cache = new_command_cache()
    command, work = cached_command(x, prev, 0, packed, cache)
    repeated, repeated_work = cached_command(x, prev, 0, packed, cache)
    np.testing.assert_array_equal(command, repeated)
    assert work > 0 and repeated_work == 0 and len(cache) == 1
    changed = x.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    cached_command(changed, prev, 0, packed, cache)
    cached_command(x, prev, 1, packed, cache)
    cached_command(x, np.full(3, .625), 0, packed, cache)
    assert len(cache) == 4
