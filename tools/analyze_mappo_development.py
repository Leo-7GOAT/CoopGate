"""Predefined conflict-stratum descriptions, never a post-hoc risk certificate."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from coopgate.confidence_v12 import conflict_group


def analyze(directory):
    directory=Path(directory)
    protocol=json.loads((directory/'protocol.json').read_text())
    if not (directory/'report.json').exists():
        raise ValueError('Complete primary evaluation first')
    rows=[json.loads(s) for s in (directory/'episodes.jsonl').read_text().splitlines()]
    lookup={(r['seed'],r['mode'],r['method']):r for r in rows}
    if len(lookup)!=protocol['expected_executions']:
        raise ValueError('Missing or duplicate observations')
    seeds=range(protocol['seed_base'],protocol['seed_base']+protocol['n_independent_initials'])
    groups={}
    for seed in seeds:
        row=lookup[seed,protocol['modes'][0],protocol['methods'][0]]
        groups[seed]=conflict_group(np.array(row['initial']))
    result={}
    for method in protocol['methods']:
        result[method]={}
        for reference in ['ego_planner','team_planner']:
            strata={}
            for group in ['all','high_conflict','low_conflict']:
                ids=[s for s in seeds if group=='all' or groups[s]==group]
                pairs=[(lookup[s,m,method],lookup[s,m,reference]) for s in ids for m in protocol['modes']]
                if not ids or any(not a['completed'] or not b['completed'] for a,b in pairs):
                    strata[group]=dict(n_independent_initials=len(ids), all_scope_gain_assessable=False)
                    continue
                delta=np.array([np.array(b['per_vehicle_cost'])-a['per_vehicle_cost'] for a,b in pairs])
                clustered=delta.reshape(len(ids),len(protocol['modes']),3).mean(1)
                lower=clustered.mean(0)-t.ppf(.95,len(ids)-1)*clustered.std(0,ddof=1)/np.sqrt(len(ids)) if len(ids)>1 else np.full(3,np.nan)
                strata[group]=dict(n_independent_initials=len(ids), repeated_mode_pairs=len(pairs),
                    all_scope_gain_assessable=True, mean_vehicle_cost_gain_s=clustered.mean(0).tolist(),
                    development_one_sided_t_lower_s=lower.tolist(),
                    ramp_worse_mode_pairs=int((delta[:,1]<-1e-9).sum()),
                    any_vehicle_cost_worse_mode_pairs=int((delta<-1e-9).any(1).sum()),
                    inference='Descriptive development strata; no multiplicity-adjusted confirmatory claim')
            result[method][reference]=strata
    report=dict(conflict_definition='Original V12 initial ETA separation <=1.25s',
                results=result, extra_simulator_executions=0, extra_independent_initials=0)
    (directory/'conflict_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(completed_methods=len(result), extra_simulator_executions=0)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory')
    analyze(parser.parse_args().directory)
