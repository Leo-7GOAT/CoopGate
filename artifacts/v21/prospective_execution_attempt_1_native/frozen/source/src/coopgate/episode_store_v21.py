"""Transactional, append-only episode blocks for long SUMO studies.

Rows and complete traces commit together. Analysis requires a sealed block and
its exact frozen schedule. An interrupted block is preserved for an explicit
recovery protocol; this class never silently resumes or replaces observations.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def encode(value):
    # Preserve the legacy JSON representation, including diagnostic infinities.
    # Statistical metric validation is separate and requires finite outcomes.
    return json.dumps(value, separators=(',', ':'), sort_keys=True).encode('utf-8')


def unpack(blob, expected):
    raw = zlib.decompress(blob)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Archived observation checksum differs')
    return json.loads(raw)


def seal_path(path):
    return Path(str(path) + '.seal.json')


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def inventory(path):
    """Integrity and case identities only; no partial effectiveness outcomes."""
    with readonly(path) as connection:
        if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite integrity check failed')
        metadata = json.loads(connection.execute('SELECT value FROM metadata WHERE name=?', ('identity',)).fetchone()[0])
        entries = list(connection.execute('SELECT position,case_key,row_sha256,trace_sha256 FROM episodes ORDER BY position'))
    if [r[0] for r in entries] != list(range(len(entries))):
        raise ValueError('Non-contiguous episode positions')
    return dict(metadata=metadata, cases=[json.loads(r[1]) for r in entries],
                row_sha256=[r[2] for r in entries], trace_sha256=[r[3] for r in entries])


class EpisodeStore:
    def __init__(self, path, metadata):
        self.path = Path(path)
        # Exclusive creation prevents both silent recovery and writer races.
        with self.path.open('xb'):
            pass
        self.connection = sqlite3.connect(self.path)
        self.connection.execute('PRAGMA journal_mode=DELETE')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE episodes(position INTEGER PRIMARY KEY,case_key TEXT UNIQUE NOT NULL,'
            'row_blob BLOB NOT NULL,trace_blob BLOB NOT NULL,row_sha256 TEXT NOT NULL,trace_sha256 TEXT NOT NULL)')
        self.connection.execute('INSERT INTO metadata VALUES(?,?)', ('identity', encode(metadata).decode()))
        self.connection.commit()
        self.position = 0

    def append(self, case, row, trace):
        if self.connection is None or seal_path(self.path).exists():
            raise ValueError('Cannot write a closed or sealed block')
        key, row_raw, trace_raw = encode(list(case)), encode(row), encode(trace)
        with self.connection:
            self.connection.execute('INSERT INTO episodes VALUES(?,?,?,?,?,?)',
                (self.position, key.decode(), zlib.compress(row_raw), zlib.compress(trace_raw),
                 hashlib.sha256(row_raw).hexdigest(), hashlib.sha256(trace_raw).hexdigest()))
        self.position += 1

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def seal(self, expected_cases):
        self.close()
        state = inventory(self.path)
        if state['cases'] != [list(c) for c in expected_cases]:
            raise ValueError('Block is incomplete or differs from its frozen schedule')
        with seal_path(self.path).open('x', encoding='utf-8') as handle:
            json.dump(dict(database_sha256=file_digest(self.path), **state), handle, indent=2)
            handle.write('\n')


def observations(path, expected_metadata, expected_cases):
    """Read only an unchanged sealed block with a complete matched schedule."""
    receipt = json.loads(seal_path(path).read_text(encoding='utf-8'))
    if (receipt['database_sha256'] != file_digest(path)
            or receipt['metadata'] != expected_metadata
            or receipt['cases'] != [list(c) for c in expected_cases]):
        raise ValueError('Sealed block identity, schedule or bytes changed')
    if inventory(path) != {k: receipt[k] for k in ('metadata', 'cases', 'row_sha256', 'trace_sha256')}:
        raise ValueError('Seal does not match stored identities and checksums')
    with readonly(path) as connection:
        query = 'SELECT case_key,row_blob,trace_blob,row_sha256,trace_sha256 FROM episodes ORDER BY position'
        for key, row_blob, trace_blob, row_sha, trace_sha in connection.execute(query):
            yield tuple(json.loads(key)), unpack(row_blob, row_sha), unpack(trace_blob, trace_sha)
