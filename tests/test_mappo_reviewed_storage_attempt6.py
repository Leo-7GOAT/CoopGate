import json
from copy import deepcopy
import pytest
from coopgate.episode_store_v21 import EpisodeStore
import coopgate.episode_store_v21 as storage
import mappo_top_two_formal_experiment as e
from mappo_reviewed_storage_attempt6 import verify_reviewed_blocks


@pytest.fixture
def block(tmp_path):
    cases=[dict(cluster_id='fixture:calibration:0',methods=['baseline','4703'])]
    plan=dict(validation_only=False,cases=dict(calibration=cases))
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    directory=tmp_path/'calibration/shard0';path=directory/'block0000';path.mkdir(parents=True)
    metadata=dict(plan_sha256=e.sha(tmp_path/'plan.json'),stage='calibration',shard=0,index=0,
                  cases_sha256=e.canonical_sha256(cases))
    keys=[[cases[0]['cluster_id'],m] for m in cases[0]['methods']]
    db=path/'observations.sqlite';store=EpisodeStore(db,metadata)
    try:
        for key in keys:store.append(key,dict(storage_fixture=True),[dict(not_a_physical_experiment=True)])
        store.seal(keys)
    finally:store.close()
    item=dict(block='block0000',metadata=metadata,keys=keys,database_sha256=e.sha(db),seal_sha256=e.sha(str(db)+'.seal.json'))
    complete=dict(blocks=[item],executions=2,expected=2)
    manifest=directory/'collection_complete.json';manifest.write_text(json.dumps(complete))
    return tmp_path,plan,complete,manifest,db


def test_complete_identity_and_every_byte_are_checked_without_redecoding(block,monkeypatch):
    root,plan,_,_,_=block
    def forbid(*args):raise AssertionError('Redundant payload decoding is not needed after native replay')
    monkeypatch.setattr(storage,'unpack',forbid)
    result=verify_reviewed_blocks(e,root,'calibration',0,plan)
    assert result['executions']==2 and result['blocks']==1
    assert result['requires_prior_actual_full_native_replay'] and not result['payloads_redecoded']


def test_database_bit_flip_is_rejected_before_any_record_can_be_accepted(block):
    root,plan,_,_,db=block
    data=bytearray(db.read_bytes());data[-1]^=1;db.write_bytes(data)
    with pytest.raises(ValueError,match='bytes changed'):
        verify_reviewed_blocks(e,root,'calibration',0,plan)


@pytest.mark.parametrize('kind',('missing','duplicate','foreign'))
def test_no_missing_duplicate_or_foreign_schedule_can_pass(block,kind):
    root,plan,complete,manifest,_=block
    value=deepcopy(complete)
    if kind=='missing':value['blocks']=[]
    elif kind=='duplicate':value['blocks']*=2
    else:value['blocks'][0]['keys'][0][0]='foreign'
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='scheduled|schedule'):
        verify_reviewed_blocks(e,root,'calibration',0,plan)


def test_forged_sealed_trace_hash_catalog_is_rejected(block):
    root,plan,complete,manifest,db=block
    seal=db.with_name(db.name+'.seal.json');receipt=json.loads(seal.read_bytes())
    receipt['trace_sha256'][0]='0'*64;seal.write_text(json.dumps(receipt))
    complete['blocks'][0]['seal_sha256']=e.sha(seal);manifest.write_text(json.dumps(complete))
    with pytest.raises(ValueError,match='hash catalog'):
        verify_reviewed_blocks(e,root,'calibration',0,plan)
