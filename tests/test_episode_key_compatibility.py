import sqlite3

import pytest

from coopgate.episode_store_v21 import EpisodeStore,observations,file_digest
from finalize_episode_keys import compatible_function,list_observations


def consume(path,metadata,schedule):
    result=[]
    for key,body,trace in observations(path,metadata,schedule):
        if key!=[body['cluster_id'],body['method']]:raise ValueError('Frozen consumer key mismatch')
        result.append((key,body,trace))
    return result


@pytest.fixture
def sealed(tmp_path):
    path=tmp_path/'observations.sqlite';metadata=dict(scope='unit-fixture-only');schedule=[['unit','baseline'],['unit','candidate']]
    store=EpisodeStore(path,metadata)
    for case in schedule:store.append(case,dict(cluster_id=case[0],method=case[1],value=1.),[dict(x=[1.,2.])])
    store.seal(schedule)
    return path,metadata,schedule


def test_actual_sealed_store_reproduces_tuple_list_failure_then_exact_payload_compatibility(sealed):
    path,metadata,schedule=sealed
    original=list(observations(path,metadata,schedule));assert isinstance(original[0][0],tuple)
    before=file_digest(path)
    with pytest.raises(ValueError,match='key mismatch'):consume(path,metadata,schedule)
    saved=consume.__globals__['observations'];adapted=compatible_function(consume)
    result=adapted(path,metadata,schedule)
    assert adapted.__code__ is consume.__code__ and consume.__globals__['observations'] is saved
    assert result==[(list(key),body,trace) for key,body,trace in original]
    assert file_digest(path)==before


def test_foreign_metadata_or_schedule_still_fail_before_yield(sealed):
    path,metadata,schedule=sealed
    with pytest.raises(ValueError):list(list_observations(path,dict(scope='foreign'),schedule))
    with pytest.raises(ValueError):list(list_observations(path,metadata,schedule[::-1]))


def test_changed_database_is_not_repaired_by_container_conversion(sealed):
    path,metadata,schedule=sealed
    with sqlite3.connect(path) as db:db.execute('UPDATE episodes SET trace_sha256=? WHERE position=0',('changed',))
    with pytest.raises(ValueError):list(list_observations(path,metadata,schedule))
