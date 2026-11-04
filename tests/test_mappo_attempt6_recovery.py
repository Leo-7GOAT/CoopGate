from pathlib import Path
import sys
import pytest
import recover_mappo_top_two_attempt6 as r


@pytest.mark.skipif(sys.platform != 'win32', reason='Exact registered Windows runtime guard is intentionally enforced')
def test_private_scopes_keep_registered_function_bodies_and_original_modules():
    original_root = r.original.FORMAL
    e, counters = r.context(0)
    assert e.FORMAL == r.OUT and r.original.FORMAL == original_root
    assert e.collect.__code__ is r.original.collect.__code__
    assert e.review.__code__ is r.original.review.__code__
    assert e.runtime.audit_record is r.original.runtime.audit_record
    assert e.runtime.make_controller is r.original.runtime.make_controller
    assert counters == dict(retained=0, new=0)


@pytest.mark.skipif(sys.platform != 'win32', reason='Exact registered Windows runtime guard is intentionally enforced')
def test_retained_partial_prefix_is_reused_without_simulation_or_mutation():
    e, counters = r.context(0)
    # A row in the unsealed original fourth block, not just an old sealed block.
    audit = r.original.read(r.AUDIT / 'report.json')
    item = next(q for q in audit['blocks'] if q['shard'] == 0 and q['index'] == 3)
    key, record, trace = next(r.observations(r.ROOT / item['cache'], item['metadata'], item['keys']))
    before = r.original.sha(r.ROOT / item['cache'])
    result = e.runtime.execute_episode(dict(cluster_id=key[0]), key[1], None, None)
    assert result == dict(record, trace=trace)
    result['trace'].clear()
    again = e.runtime.execute_episode(dict(cluster_id=key[0]), key[1], None, None)
    assert again['trace'] == trace
    assert counters == dict(retained=2, new=0)
    assert r.original.sha(r.ROOT / item['cache']) == before


def test_native_redirection_cannot_launch_the_old_output_root():
    for name in ('mappo_top_two_formal_experiment.py', 'mappo_top_two_formal_finalize.py'):
        args = r.redirect(r.ROOT / 'tools' / name, 'review')
        assert str(Path(r.__file__).resolve()) in args and args[-1] == 'review'
        assert str(r.ROOT / 'tools' / name) not in args
    with pytest.raises(ValueError):
        r.redirect(r.ROOT / 'tools/unregistered.py')
