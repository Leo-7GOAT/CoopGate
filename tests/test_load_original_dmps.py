from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import load_original_dmps as loader


def test_modified_upstream_file_rejected_before_import(tmp_path):
    (tmp_path/'mcts.py').write_text('raise RuntimeError("must not execute")\n')
    with pytest.raises(ValueError,match='Original author module changed'):
        loader.load(tmp_path)
