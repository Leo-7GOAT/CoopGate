"""Unchanged pure helpers extracted from the original formal implementation."""
import hashlib
import json
import time
import numpy as np
from coopgate.shifted_sumo_v19 import shifted_initial

def canonical_sha256(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def array_sha256(values, *, decimals=None):
    array = np.asarray(values, dtype='<f8')
    if decimals is not None:
        array = np.round(array, decimals=decimals)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order='C')).hexdigest()


def planned_initial(seed, scenario):
    if scenario in ('nominal', 'dense'):
        rng = np.random.default_rng(seed)
        velocity = rng.uniform(8., 14., 3) if scenario == 'nominal' else rng.uniform(12., 17., 3)
        position = -rng.uniform(70., 110., 3) if scenario == 'nominal' else -rng.uniform(70., 85., 3)
        position[2] = position[0]-rng.uniform(30., 45.)
        return np.column_stack((position, velocity))
    return shifted_initial(seed, scenario)


def json_ready(value):
    return json.loads(json.dumps(value, allow_nan=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else item.tolist()))


class TimedGate:
    """Keep the gate's evidence unchanged; include unsuccessful select calls."""
    def __init__(self, gate):
        self.gate = gate
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.gate, name)

    def select(self, *args, **kwargs):
        started = time.perf_counter()
        error = None
        try:
            return self.gate.select(*args, **kwargs)
        except BaseException as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            self.calls.append(dict(wall_ms=1000*(time.perf_counter()-started), error=error))


def check_calls(row, trace, calls):
    successes = [c for c in calls if c['error'] is None]
    failures = [c for c in calls if c['error'] is not None]
    if (len(successes) != len(trace) or len(failures) != int(row['aborted'])
            or (failures and calls[-1] is not failures[-1])
            or any(not np.isfinite(c['wall_ms']) or c['wall_ms'] < 0 for c in calls)):
        raise ValueError('Incomplete successful/aborted select timing')
    if failures and failures[0]['error']['message'] != row['abort_reason']:
        raise ValueError('Terminal select error differs from recorded abort')
