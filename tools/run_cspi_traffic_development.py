"""Episode-level CSPI traffic integration on an explicitly reused development grid."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run_cspi_author_pipeline as author

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'artifacts/v38/interrupted_block_grid_recovery/v36/analysis/report.json'
REPORT_SHA = '1e13918534046bc71b3a39a0ee340c5852357778930dc5cbc8b96cf4eb175162'
MODELS = ('1802', '1812', '1822')
PLANTS = ('nominal', 'gain_drag', 'lag_mild', 'lag_strong')
BASELINE = 1e6


def committed(name):
    raw = subprocess.run(['git', 'show', 'HEAD:'+name], cwd=ROOT, check=True, capture_output=True).stdout
    if raw != (ROOT/name).read_bytes():
        raise ValueError('Exact protocol and implementation must be committed before execution: '+name)


def observations(features, treatment, observed):
    """Fit only supplied observed rewards; no alternative branch enters this API."""
    scaler = StandardScaler().fit(features[:64])
    x = scaler.transform(features)
    predictions, parameters = [], []
    for arm in (0, 1):
        mask = treatment[:64] == arm
        if mask.sum() < 8:
            raise ValueError('Insufficient fixed training assignment; do not resample')
        model = Ridge(alpha=1.).fit(x[:64][mask], observed[:64][mask])
        predictions.append(model.predict(x))
        parameters.append(dict(arm=arm, training_count=int(mask.sum()),
                               coef=model.coef_.tolist(), intercept=float(model.intercept_)))
    m0, m1 = predictions
    frame = pd.DataFrame(dict(treatment=treatment, propensity=np.full(160, .5), outcome=observed,
                              mu_0_est=m0, mu_1_est=m1, ite_est=m1-m0))
    return frame, dict(models=parameters, scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist())


def estimate_checks(f, data, cutoffs):
    if np.isin(data.ite_est.to_numpy(), np.append(cutoffs, BASELINE)).any():
        raise ValueError('Original PDE threshold endpoint ambiguity')
    author.verify_estimator(f, data, cutoffs, BASELINE)
    values, covariance = f['PDE'](cutoffs, BASELINE, data, by_quantile=False, dr=True)
    if not np.isfinite(values).all() or not np.isfinite(covariance).all() or np.any(np.diag(covariance) <= 0):
        raise ValueError('Nonfinite or zero-variance candidate; preserve untestable result')


def select(f, data, index):
    scores = data.ite_est.to_numpy()
    if not np.isfinite(scores).all() or np.any(scores >= BASELINE):
        raise ValueError('Score outside fixed baseline threshold domain')
    cutoffs = np.quantile(scores[:64], [.1, .3, .5, .7, .9])
    if len(set(cutoffs)) != 5:
        raise ValueError('Duplicate fixed training-quantile candidates')
    tune, test = data.iloc[64:96], data.iloc[96:128]
    estimate_checks(f, tune, cutoffs)
    estimate_checks(f, test, cutoffs)
    np.random.seed(804620+index)
    single, single_dr, mt_e, mt = f['cutoff_selection'](cutoffs, BASELINE, tune,
                                                      gamma=.05, n_sim=10000, test_ratio=1)
    outputs = []
    for name, selected in [('cspi', single_dr[1:2]), ('cspi_mt', mt)]:
        selected = np.asarray(selected, dtype=float)
        if name == 'cspi':
            passed = f['safety_test_single'](selected, BASELINE, test, gamma=.05, dr=True)
        else:
            np.random.seed(804630+index)
            passed = f['safety_test'](selected, BASELINE, test, gamma=.05, nsim=10000, dr=True)
        passed = np.asarray(passed, dtype=bool).reshape(-1)
        assert selected.shape == passed.shape and np.isfinite(selected).all()
        candidates = selected[passed]
        chosen = BASELINE
        if len(candidates):
            values, _ = f['PDE'](candidates, BASELINE, pd.concat([tune, test]), by_quantile=False, dr=True)
            chosen = float(candidates[np.argmax(values)])
        outputs.append(dict(method=name, status='tested', selected=selected.tolist(),
                            passed=passed.tolist(), deployed_cutoff=chosen))
    return dict(candidates=cutoffs.tolist(), methods=outputs,
                original_selection_outputs=[np.asarray(x).tolist() for x in (single, single_dr, mt_e, mt)])


def evaluate(cases, choices):
    selected = [r if choice else dict(row=r['baseline'], baseline=r['baseline'], adopted=False,
                                     controller_failed=r['baseline_failed'],
                                     baseline_failed=r['baseline_failed'], relative_harm=None if r['baseline_failed'] else False)
                for r, choice in zip(cases, choices)]
    assert len(selected) == 32
    failed = sum(bool(r['controller_failed']) for r in selected)
    base_failed = sum(bool(r['baseline_failed']) for r in selected)
    adoption = sum(bool(r['adopted']) for r in selected)
    harms = sum(bool(r['adopted'] and r['relative_harm']) for r in selected)
    before = sum(r['baseline']['team_cost'] for r in selected)
    after = sum(r['row']['team_cost'] for r in selected)
    return dict(runs=32, controller_selected=int(sum(choices)), actual_adopted=adoption,
                harmful_and_adopted=harms, completed=32-failed, baseline_failed=base_failed,
                empirical_conditional_harm=harms/adoption if adoption else None,
                all_scope_cost_gain_percent=100*(before-after)/before if not failed and not base_failed else None)


def run(source, output):
    for name in ('tools/run_cspi_traffic_development.py', 'docs/cspi_traffic_development_protocol.md',
                 'tools/run_cspi_author_pipeline.py'):
        committed(name)
    if author.digest(REPORT) != REPORT_SHA:
        raise ValueError('Completed development grid changed')
    f = author.functions(source)
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    records = {tuple(r['case']): r for r in report['cases']}
    units = sorted({key[:4] for key in records})
    assert len(units) == 160 and len(records) == 8320
    rng = np.random.default_rng(804600)
    plants = rng.integers(0, 4, 160)
    treatment = rng.binomial(1, .5, 160)
    order = rng.permutation(160)
    keys = [units[i]+(PLANTS[plants[i]],) for i in order]
    treatment = treatment[order]
    splits = ['fit']*64+['tune']*32+['safety_test']*32+['evaluation']*32
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(source_report_sha256=REPORT_SHA, upstream_commit=author.COMMIT,
                    upstream_git_blob=author.BLOB, upstream_sha256=author.digest(source),
                    driver_sha256=author.digest(Path(__file__)),
                    protocol_sha256=author.digest(ROOT/'docs/cspi_traffic_development_protocol.md'),
                    randomization_seed=804600, new_sumo_runs=0, reused_initials=160,
                    fresh_independent_initials=0, new_risk_certificate=False,
                    runtime={p:author.importlib.metadata.version(p) for p in
                             ('numpy','scipy','pandas','scikit-learn','threadpoolctl')},
                    assignments=[dict(condition=list(key), treatment=int(t), split=s)
                                 for key,t,s in zip(keys,treatment,splits)])
    author.write(output/'protocol.json', protocol)
    results = []
    for index, model in enumerate(MODELS):
        cases = [records[key+('nominal_performance', model)] for key in keys]
        banks = [records[key+('bank_performance', model)] for key in keys]
        assert all(not r['controller_failed'] and not r['baseline_failed'] for r in cases+banks)
        features = np.array([np.asarray(r['row']['initial']).ravel().tolist()
                             +[float(r['row']['plant_name']==p) for p in PLANTS] for r in cases])
        y0 = -np.array([r['baseline']['team_cost'] for r in cases])
        y1 = -np.array([r['row']['team_cost'] for r in cases])
        observed = np.where(treatment, y1, y0)
        data, parameters = observations(features, treatment, observed)
        # Separate hidden potential rewards from the actual function inputs.
        np.savez_compressed(output/(model+'_observed.npz'), features=features,
                            **{c:data[c].to_numpy() for c in data.columns})
        np.savez_compressed(output/(model+'_audit_potential_rewards.npz'), y0=y0, y1=y1)
        try:
            selection = select(f, data, index)
        except ValueError as error:
            selection = dict(status='untestable', reason=str(error), methods=[
                dict(method=m, status='untestable', deployed_cutoff=BASELINE) for m in ('cspi','cspi_mt')])
        for method in selection['methods']:
            choices = data.ite_est.to_numpy()[128:] >= method['deployed_cutoff']
            method['evaluation'] = evaluate(cases[128:], choices)
        result = dict(model=model, nuisance_parameters=parameters, selection=selection,
                      comparisons=dict(always_baseline=evaluate(cases[128:], np.zeros(32, bool)),
                                       always_nominal=evaluate(cases[128:], np.ones(32, bool)),
                                       always_bank=evaluate(banks[128:], np.ones(32, bool))))
        author.write(output/(model+'_result.json'), result)
        results.append(result)
        print(json.dumps(dict(model=model, selection=selection, comparisons=result['comparisons'])), flush=True)
    author.write(output/'complete.json', dict(models=3, original_units=160, evaluation_units=32,
        fresh_independent_initials=0, new_sumo_runs=0, new_risk_certificate=False,
        all_models_testable=all(r['selection'].get('status')!='untestable' for r in results),
        files_sha256={p.name:author.digest(p) for p in output.iterdir() if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.output)
