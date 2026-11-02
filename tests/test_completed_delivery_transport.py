"""Actual local-Git transfer verification;no network or scientific samples."""
from pathlib import Path
import sys
import subprocess
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from publish_completed_delivery import publish


def test_exact_tree_dirty_bytes_and_bounded_fast_forward_parts(tmp_path):
    remote=tmp_path/'remote.git';repo=tmp_path/'work';repo.mkdir()
    def git(*a):return subprocess.check_output(['git',*a],cwd=repo,text=True,stderr=subprocess.DEVNULL).strip()
    subprocess.run(['git','init','--bare',str(remote)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    git('init','-b','main');git('config','user.name','Local transport validation');git('config','user.email','local@example.invalid')
    git('remote','add','origin',str(remote))
    (repo/'notes.txt').write_bytes(b'original notes\r\n');(repo/'remove.txt').write_bytes(b'obsolete\n')
    git('add','.');git('commit','-m','base');base=git('rev-parse','HEAD');git('push','origin','main')
    for index in range(3):
        with zipfile.ZipFile(repo/f'data{index}.zip','x') as z:z.writestr('data.bin',bytes(range(256))*(index+1))
    (repo/'results.json').write_text('{"example":true}\n');(repo/'remove.txt').unlink()
    git('add','-A');git('commit','-m','exact completed artifacts');original=git('rev-parse','HEAD');tree=git('write-tree')
    dirty=b'preserve this uncommitted change\r\n';(repo/'notes.txt').write_bytes(dirty)
    state=publish(repo,base,original,'test-transport',str(remote))
    assert state['completed'] and state['final_tree_identical'] and len(state['parts'])==4
    assert (repo/'notes.txt').read_bytes()==dirty and git('write-tree')==tree
    assert git('rev-parse','HEAD^{tree}')==tree and git('rev-parse','backup-test-transport')==original
    assert git('ls-remote','origin','refs/heads/main').split()[0]==state['published_head']
    previous=base
    for part in state['parts']:
        assert git('rev-parse',part['commit']+'^')==previous
        assert part['remote_confirmed'] and part['attempts'][-1]['actual_native_exit_code']==0
        previous=part['commit']
    assert 'notes.txt' in git('diff','--name-only') and not git('diff','--cached','--name-only')
