"""Measure cold imports/JIT in a fresh cache; never include it in warm latency."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True); args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('Refusing startup measurement overwrite')
    args.cache_dir.mkdir(parents=True, exist_ok=False)
    os.environ['NUMBA_CACHE_DIR'] = str(args.cache_dir.resolve())
    start = time.perf_counter()
    import numpy as np
    import numba
    from coopgate.fast_v8 import plan_fast
    from coopgate.merge_v4 import MergeConfig
    imports_ms = (time.perf_counter()-start)*1000
    initial = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    kernel_start = time.perf_counter(); plan_fast(initial, np.zeros(3), MergeConfig(), pruning=True)
    kernel_ms = (time.perf_counter()-kernel_start)*1000
    warm_start = time.perf_counter(); plan_fast(initial, np.zeros(3), MergeConfig(), pruning=True)
    warm_ms = (time.perf_counter()-warm_start)*1000
    result = {'imports_ms': imports_ms, 'first_kernel_call_ms': kernel_ms, 'second_kernel_call_ms': warm_ms,
              'numba_version': numba.__version__, 'cache_was_empty': True,
              'scope': 'One new Python process, fresh cache; cold test excludes OS process launch but includes package imports and compilation. No control should start before warmup.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n', newline='\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
