"""Publish scope tables/figure only after the original complete OOD replay exits zero."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/recovery_20261026/ood_recovery'
OUT=ROOT/'artifacts/recovery_20261027/ood_scope_summary'
JOURNAL=ROOT/'.git/mainline-20261027.jsonl'
PLANTS=('delay_gain_lag_mild','delay_gain_lag_heterogeneous','delay_gain_lag_severe')
PERCEPTIONS=('clean_reference','bounded_bias_jitter','stale_dropout')
MODELS=('1802','1812','1822')
METHODS=(('baseline','none'),*((rule,model) for rule in
         ('nominal_performance','bank_performance','mps_shared','mamps_shared') for model in MODELS))


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def committed(path):
    name=path.relative_to(ROOT).as_posix()
    data=subprocess.check_output(['git','cat-file','blob','HEAD:'+name],cwd=ROOT)
    if hashlib.sha256(data).hexdigest()!=digest(path):
        raise ValueError('Complete committed report required: '+name)


def read_complete():
    events=[json.loads(line) for line in JOURNAL.read_text(encoding='utf-8').splitlines()]
    if not any(e['kind']=='owned_process_exit' and e['detail'].get('name')=='ood_summary'
               and e['detail'].get('exit_code')==0 for e in events):
        raise ValueError('No actual zero-exit receipt for the full original replay; no outcomes read')
    report=SOURCE/'analysis/report.json'
    checksum=SOURCE/'analysis/sha256.json'
    committed(report);committed(checksum)
    if digest(report)!=json.loads(checksum.read_text(encoding='utf-8'))['report.json']:
        raise ValueError('Original completed report checksum differs')
    value=json.loads(report.read_text(encoding='utf-8'))
    if (value['actual_sumo_runs']!=22464 or value['paired_condition_blocks']!=1728
            or value['independent_initials']!=192
            or not value['all_episodes_independently_replayed']
            or not value['all_fallback_identities_exact']
            or value['new_formal_safety_certificate']):
        raise ValueError('Full original scope and claim boundaries are required')
    if len(value['cells'])!=351 or sum(c['episodes'] for c in value['cells'])!=22464:
        raise ValueError('Missing repeated factor cells')
    if any(c['episodes']!=64 or not 0<=c['failures']<=64
           or not 0<=c['adopted']<=64 for c in value['cells']):
        raise ValueError('Unexpected cell sample size or counts')
    return value


def main():
    value=read_complete()
    if OUT.exists():raise FileExistsError('Preserve existing complete scope summary')
    factors=[(plant,perception) for plant in PLANTS for perception in PERCEPTIONS]
    descriptive=[]
    for rule,model in METHODS:
        for plant,perception in factors:
            cells=[c for c in value['cells'] if (c['rule'],c['model'],c['plant'],c['perception'])
                   ==(rule,model,plant,perception)]
            if len(cells)!=3 or len({c['regime'] for c in cells})!=3:
                raise ValueError('All three predeclared traffic regimes required')
            n=sum(c['episodes'] for c in cells)
            descriptive.append(dict(rule=rule,model=model,plant=plant,perception=perception,
                episodes=n,failures=sum(c['failures'] for c in cells),
                adopted_episodes=sum(c['adopted'] for c in cells),
                failure_percent=100*sum(c['failures'] for c in cells)/n,
                adopted_episode_percent=100*sum(c['adopted'] for c in cells)/n))
    primary=[dict(model=model,**value['primary']['models'][model]) for model in MODELS]
    OUT.mkdir(parents=True)
    for name,rows in (('factor_cells.csv',descriptive),('original_primary_decisions.csv',primary)):
        with (OUT/name).open('x',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    matrix=np.array([r['failure_percent'] for r in descriptive]).reshape(13,9)
    fig,ax=plt.subplots(figsize=(7.3,5.2),layout='constrained')
    image=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlOrRd',aspect='auto')
    ax.set_xticks(range(9),[f'{p}\n{s}' for p in ('mild','hetero.','severe')
                          for s in ('clean','bias','stale')],fontsize=6.8)
    ax.set_yticks(range(13),[('baseline' if m=='none' else f'{r}/{m}') for r,m in METHODS],fontsize=6.8)
    for y in range(13):
        for x in range(9):
            ax.text(x,y,f'{matrix[y,x]:.1f}',ha='center',va='center',fontsize=6.5,
                    color='white' if matrix[y,x]>=60 else 'black')
    for x in (2.5,5.5):ax.axvline(x,color='0.4',linewidth=.8)
    ax.set_title('Empirical episode failure (%) under compound mismatch',fontsize=10)
    ax.set_xlabel('bias: bounded bias/jitter; stale: dropout/staleness\n'
                  '192 initial conditions per cell; the same initials repeat across nine cells',fontsize=7)
    fig.colorbar(image,ax=ax,label='Failure (%)',shrink=.85)
    fig.savefig(OUT/'failure_scope.png',dpi=300)
    fig.savefig(OUT/'failure_scope.svg')
    plt.close(fig)
    summary=dict(source_report_sha256=digest(SOURCE/'analysis/report.json'),
        actual_sumo_runs=22464,independent_initials=192,repeated_factor_cells_per_initial=9,
        all_three_fixed_models_pass=value['primary']['all_three_fixed_models_pass'],
        original_primary=value['primary'],new_inferential_tests=0,new_sumo_runs=0,
        new_population_safety_certificate=False,failures_omitted=0,
        notes=[
            'The original Holm-adjusted cluster-bootstrap decisions are copied unchanged.',
            'Original percentile bounds are descriptive and unadjusted; no finite-sample population-risk guarantee follows.',
            'The heatmap is descriptive; nine repeated cells do not create nine independent samples per initial.',
            'mps_shared and mamps_shared are project adapters, not full original-author reproductions.',
            'Failure-penalized cost gain is not a failure-free travel-time efficiency gain.',
            'No controller latency ranking or communication-delay statistic is produced.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    files={p.name:digest(p) for p in OUT.iterdir() if p.is_file()}
    (OUT/'sha256.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(all_three_fixed_models_pass=summary['all_three_fixed_models_pass'],
                         artifacts=str(OUT),new_inferential_tests=0)),flush=True)


if __name__=='__main__':main()
