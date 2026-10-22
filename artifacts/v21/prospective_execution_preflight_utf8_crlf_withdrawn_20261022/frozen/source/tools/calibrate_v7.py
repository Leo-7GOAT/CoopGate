"""Freeze simultaneous conditional-risk threshold before test execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.learning_v6 import upper_quantile
from coopgate.risk_v7 import calibrate_threshold

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v7'


def main():
    path = D/'calibration.json'
    if path.exists():
        raise RuntimeError('Refusing recalibration')
    protocol = json.loads((D/'protocol.json').read_text())
    groups = {s: dict(np.load(D/f'data/{s}.npz')) for s in protocol['calibration_seeds']}
    pac = calibrate_threshold(groups, protocol['thresholds_s'], protocol['risk'], protocol['delta'], protocol['tolerance_s'])
    quantiles = {s: upper_quantile(v['actual']-v['predicted'], protocol['risk']) for s, v in groups.items()}
    sources = [D/'protocol.json', *[D/f'data/{s}.npz' for s in groups]]
    result = {'pac': pac, 'conformal_quantiles_s': quantiles, 'max_conformal_quantile_s': max(quantiles.values()),
              'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    path.write_text(json.dumps(result, indent=2)+'\n', newline='\n')
    print(json.dumps({'pac_selected': pac['selected'], 'conformal_quantiles': quantiles}, indent=2))


if __name__ == '__main__':
    main()
