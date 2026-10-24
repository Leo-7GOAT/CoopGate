"""Diagnose fixed traffic candidate support and literal upstream behavior.

Post-result diagnosis only: no thresholds, samples, selections or deployment
results from the registered integration are replaced by this output.
"""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_cspi_author_pipeline as author


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def diagnose(source, root):
    f = author.functions(source)
    complete = json.loads((root/'complete.json').read_text(encoding='utf-8'))
    for name, sha in complete['files_sha256'].items():
        assert author.digest(root/name) == sha
    results = []
    for index, model in enumerate(('1802', '1812', '1822')):
        arrays = np.load(root/(model+'_observed.npz'))
        data = pd.DataFrame({k:arrays[k] for k in arrays.files if k != 'features'})
        cutoffs = np.quantile(data.ite_est[:64], [.1,.3,.5,.7,.9])
        tune, test = data.iloc[64:96], data.iloc[96:128]
        result = dict(model=model, cutoffs=cutoffs, support={})
        for name, subset in [('tune',tune), ('safety_test',test)]:
            author.verify_estimator(f, subset, cutoffs, 1e6)
            values, covariance = f['PDE'](cutoffs, 1e6, subset, by_quantile=False, dr=True)
            result['support'][name] = dict(selected_counts=[int((subset.ite_est>=c).sum()) for c in cutoffs],
                                           estimates=values, variance=np.diag(covariance))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                np.random.seed(804620+index)
                selections = f['cutoff_selection'](cutoffs,1e6,tune,gamma=.05,n_sim=10000,test_ratio=1)
                result['literal_upstream_selection_outputs'] = selections
                single = f['safety_test_single'](selections[1][1:2],1e6,test,gamma=.05,dr=True)
                np.random.seed(804630+index)
                multiple = f['safety_test'](selections[3],1e6,test,gamma=.05,nsim=10000,dr=True)
                result['literal_upstream_test_passes'] = dict(cspi=single,cspi_mt=multiple)
            except Exception as error:
                result['literal_upstream_exception'] = dict(type=type(error).__name__,message=str(error))
            result['warnings'] = [dict(category=w.category.__name__,message=str(w.message)) for w in captured]
        results.append(result)
    return finite_json(dict(purpose='Post-result zero-support diagnosis, not replacement or comparative efficacy',
                            source_sha256=author.digest(source),
                            input_completion_sha256=author.digest(root/'complete.json'), models=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = diagnose(args.source,args.input)
        author.write(args.output,result)
        print(json.dumps(result))
