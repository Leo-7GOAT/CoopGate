from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_mappo_recovery import schedule


def test_all_recovery_classes_and_modes_are_paired_without_inflating_initials():
    cases=schedule(2,96800001,'integration')
    assert len(cases)==160
    assert len(set(cases))==160
    assert len({(k[0],k[1]) for k in cases})==10
    for identity in {(k[0],k[1]) for k in cases}:
        assert len([k for k in cases if k[:2]==identity])==16
