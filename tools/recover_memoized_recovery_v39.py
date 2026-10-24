"""Corrected, evidence-bound retry of the interrupted V37 memoization audit.

V38 is retained permanently as a pre-simulation failed attempt.  This module
configures the audited recovery mechanics to write only to a fresh V39 output
directory and binds both the original V37 prefix and the complete V38 failure
receipt before any simulator process may start.
"""
import argparse
from pathlib import Path

import recover_memoized_recovery_v38 as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v39/memoized_recovery_recovery_20261024'
FAILED_ATTEMPT = ROOT/'artifacts/v38/memoized_recovery_recovery_20261024'
SCRIPT = 'tools/recover_memoized_recovery_v39.py'
TEST = 'tests/test_recover_memoized_recovery_v39.py'


def configure():
    core.OUT = OUT
    core.SCRIPT = SCRIPT
    core.TEST = TEST
    core.SOURCES = (
        'tools/recover_memoized_recovery_v38.py',
        'tests/test_recover_memoized_recovery_v38.py',
        SCRIPT,
        TEST,
    )
    core.FAILED_ATTEMPT = FAILED_ATTEMPT
    core.RECOVERY_SEGMENT = 'recovered_v39'
    return core


def freeze():
    configure().freeze()


def worker():
    configure().worker()


def watch():
    configure().watch()


def summarize():
    configure().summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','worker','watch','summarize'))
    globals()[parser.parse_args().command]()
