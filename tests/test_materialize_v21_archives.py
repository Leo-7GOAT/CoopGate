import hashlib
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from materialize_v21_archives import confined, restore_archive


def make_archive(tmp_path, name='trace.json', content=b'{"failed": true}\n'):
    archive = tmp_path/'original.zip'
    with zipfile.ZipFile(archive, 'x') as target:
        target.writestr(name, content)
    return archive, {name:hashlib.sha256(content).hexdigest()}


def test_restoration_is_byte_exact_exclusive_and_repeatable(tmp_path):
    archive, members = make_archive(tmp_path, 'phase/trace.json')
    target = tmp_path/'restored'
    assert restore_archive(archive, target, members, check_only=True)['missing'] == 1
    assert not target.exists()
    assert restore_archive(archive, target, members)['restored'] == 1
    assert (target/'phase/trace.json').read_bytes() == b'{"failed": true}\n'
    assert restore_archive(archive, target, members)['restored'] == 0
    (target/'phase/trace.json').write_bytes(b'user change')
    with pytest.raises(ValueError):
        restore_archive(archive, target, members)
    assert (target/'phase/trace.json').read_bytes() == b'user change'


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'C:/outside', 'phase\\..\\outside'])
def test_archive_paths_cannot_leave_the_destination(tmp_path, name):
    with pytest.raises(ValueError):
        confined(tmp_path, name)


def test_bad_bytes_do_not_publish_a_corrupt_trace(tmp_path):
    archive, members = make_archive(tmp_path)
    members['trace.json'] = '0'*64
    target = tmp_path/'restored'
    with pytest.raises(ValueError):
        restore_archive(archive, target, members)
    assert not (target/'trace.json').exists()
    assert not list(target.glob('*.partial'))


def test_an_unregistered_archive_member_cannot_be_extracted(tmp_path):
    archive, _ = make_archive(tmp_path)
    with pytest.raises(ValueError):
        restore_archive(archive, tmp_path/'restored', {})
