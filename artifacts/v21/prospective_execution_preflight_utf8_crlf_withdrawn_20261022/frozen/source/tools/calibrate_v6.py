"""Fix one-sided performance threshold after model/validation selection."""
import hashlib
import json
from pathlib import Path

import numpy as np

from coopgate.learning_v6 import load_models, predict, upper_quantile


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v6'


def main():
    path = D/'calibration.json'
    if path.exists():
        raise RuntimeError('Refusing recalibration')
    protocol = json.loads((D/'protocol.json').read_text())
    models = load_models(D, protocol['training']['seeds'])
    cal = np.load(D/'data/calibration.npz')
    action, predicted = predict(models, cal['states'])
    actual = cal['costs'][np.arange(len(action)), action]-cal['costs'][:, 0]
    residuals = actual-predicted
    alpha = protocol['calibration_alpha']
    q = upper_quantile(residuals, alpha)
    result = {'n': len(action), 'alpha': alpha, 'upper_residual_quantile_s': q,
              'scope': 'Marginal upper excess-cost coverage for nominal exchangeable scenes only; not conditional acceptance or road safety.',
              'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [D/'data/calibration.npz', *[D/f'seed{s}/policy.pt' for s in protocol['training']['seeds']]]},
              'actions': action.tolist(), 'predicted_excess_s': predicted.tolist(),
              'actual_excess_s': actual.tolist(), 'residuals_s': residuals.tolist()}
    path.write_text(json.dumps(result, indent=2)+'\n', newline='\n')
    print(json.dumps({'n': len(action), 'q_s': q, 'calibration_acceptance': float(np.mean(predicted+q <= 0))}))


if __name__ == '__main__':
    main()
