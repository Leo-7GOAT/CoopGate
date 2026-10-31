"""Delivery integrity tests use synthetic bytes only; no simulation outcomes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location(
    'joint_delivery', Path(__file__).resolve().parents[1] /
    'operations/archive_completed_joint_20261031.py')
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


class CompletedJointDeliveryTests(unittest.TestCase):
    def test_multivolume_bytes_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root / 'source', root / 'archive'
            source.mkdir()
            files = []
            for name, body in [('a.json', b'{"certified":false}\r\n'),
                               ('b.sqlite', bytes(range(120)))]:
                path = source / name
                path.write_bytes(body)
                files.append(path)
            result = delivery.archive_bytes(source, target, files, limit=125)
            self.assertEqual(len(result['archive_parts']), 2)
            restored = root / 'restored'
            for part in result['archive_parts']:
                with zipfile.ZipFile(target / part['name']) as archive:
                    archive.extractall(restored)
            for path in files:
                self.assertEqual(path.read_bytes(), (restored / path.name).read_bytes())
            with self.assertRaises(FileExistsError):
                delivery.archive_bytes(source, target, files, limit=125)

    def test_incomplete_finalizer_refuses_before_any_archive_or_outcome_read(self):
        with patch.object(delivery, 'require_finalization',
                          side_effect=ValueError('not complete')), \
                patch.object(delivery, 'source_files') as files, \
                patch.object(delivery, 'archive_bytes') as archive, \
                patch.object(delivery, 'TARGET', Path('nonexistent-delivery-test-target')):
            with self.assertRaisesRegex(ValueError, 'not complete'):
                delivery.main()
            files.assert_not_called()
            archive.assert_not_called()

    def test_oversized_file_refuses_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            path = source / 'large'
            path.write_bytes(b'abcd')
            target = root / 'archive'
            with self.assertRaisesRegex(ValueError, 'exceeds delivery budget'):
                delivery.archive_bytes(source, target, [path], limit=3)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
