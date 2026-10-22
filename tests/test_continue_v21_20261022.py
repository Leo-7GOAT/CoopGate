from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from continue_v21_20261022 import split_prefix


def test_only_the_exact_missing_suffix_is_executed():
    expected=[(i,'plant','method') for i in range(5)]
    assert split_prefix(expected,expected[:3])==expected[3:]
    assert split_prefix(expected,expected)==[]


@pytest.mark.parametrize('present', [[(1,'plant','method')],[(0,'plant','method')]*2,
                                    [(0,'plant','method'),(2,'plant','method')],[(0,'changed','method')]])
def test_recovery_cannot_skip_duplicate_relabel_or_replace_prior_cases(present):
    with pytest.raises(ValueError):split_prefix([(i,'plant','method') for i in range(5)],present)
