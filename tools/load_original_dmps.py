"""Load pinned author modules from a user-supplied checkout without vendoring."""
import hashlib
import importlib.util
from pathlib import Path
import sys

COMMIT='f37df2d3900cfdc330ebb39e9b89d05dd9b85a7e'
BLOBS={'mcts.py':'68a83a6eaf518cc4ed7d2f07ccf7cd7f7ccc686c',
       'mcts_trainer.py':'34afb3aefdaee04a8b7228d0e15adb1e2fdb6f10',
       'models.py':'2fa6a782bae306945ffe8fe8af18e9b94121149a',
       'utils.py':'f2a00884e6dcf3301154e7b28e8891c2b1df8168'}


def verify(root):
    result={}
    for name,blob in BLOBS.items():
        raw=(Path(root)/name).read_bytes()
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=blob:
            raise ValueError('Original author module changed: '+name)
        result[name]=dict(git_blob=blob,sha256=hashlib.sha256(raw).hexdigest())
    return result


def load(root):
    metadata=verify(root)
    def module(stem):
        name='coopgate_reference_dmps_'+stem
        spec=importlib.util.spec_from_file_location(name,Path(root)/(stem+'.py'))
        result=importlib.util.module_from_spec(spec);sys.modules[name]=result
        spec.loader.exec_module(result)
        return result
    dependencies={name:module(name) for name in ('models','utils','mcts')}
    absent=object();saved={name:sys.modules.get(name,absent) for name in dependencies}
    try:
        # Restore generic import aliases immediately after importing the trainer;
        # its bound Model/polyak_update/MCTSPlanner retain the author objects.
        sys.modules.update(dependencies)
        trainer=module('mcts_trainer')
    finally:
        for name,value in saved.items():
            if value is absent:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    return trainer,metadata
