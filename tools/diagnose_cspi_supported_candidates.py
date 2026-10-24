"""Post-result development diagnostic; not a replacement for fresh CSPI evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import cspi_supported_candidates as adapter
import run_cspi_author_pipeline as author
from run_cspi_traffic_development import committed

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('tools/cspi_supported_candidates.py','tools/diagnose_cspi_supported_candidates.py',
         'tests/test_cspi_supported_candidates.py','docs/cspi_supported_candidates_protocol.md',
         'tools/run_cspi_author_pipeline.py','tools/run_cspi_traffic_development.py')


def run(source, parent, output):
    for name in SOURCES:
        committed(name)
    hashes={name:author.digest(ROOT/name) for name in SOURCES}
    complete=json.loads((parent/'complete.json').read_text(encoding='utf-8'))
    original_hash=author.digest(parent/'complete.json')
    for name,sha in complete['files_sha256'].items():
        if author.digest(parent/name)!=sha:
            raise ValueError('Old development input changed')
    f=author.functions(source)
    output.mkdir(parents=True,exist_ok=False)
    author.write(output/'protocol.json',dict(source_sha256=hashes,input_complete_sha256=original_hash,
        upstream_sha256=author.digest(source),known_prior_test_outcomes=True,new_sumo_runs=0,
        new_independent_initials=0,new_risk_certificate=False,
        runtime={p:author.importlib.metadata.version(p) for p in ('numpy','scipy','pandas','scikit-learn','threadpoolctl')}))
    results=[]
    for index,model in enumerate(('1802','1812','1822')):
        with np.load(parent/(model+'_observed.npz')) as archive:
            data=pd.DataFrame({k:archive[k] for k in archive.files if k!='features'})
        selection=adapter.choose(f,data.ite_est.to_numpy()[:64],data.iloc[64:96].copy(),804620+index)
        author.write(output/(model+'_selection.json'),selection)
        # Read back the persisted selection before passing only the safety rows.
        frozen=json.loads((output/(model+'_selection.json')).read_text(encoding='utf-8'))
        result=adapter.test_selected(f,frozen,data.iloc[96:128].copy(),804630+index)
        author.write(output/(model+'_test.json'),result)
        results.append(dict(model=model,retained=selection['payload']['retained'],**result))
    for name,sha in hashes.items():
        if author.digest(ROOT/name)!=sha:
            raise ValueError('Source changed during diagnostic')
    for name,sha in complete['files_sha256'].items():
        if author.digest(parent/name)!=sha:
            raise ValueError('Input changed during diagnostic')
    if author.digest(parent/'complete.json')!=original_hash:
        raise ValueError('Input manifest changed')
    author.write(output/'complete.json',dict(models=results,new_sumo_runs=0,new_independent_initials=0,
        new_risk_certificate=False,known_prior_test_outcomes=True,
        files_sha256={p.name:author.digest(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source,args.input,args.output)
