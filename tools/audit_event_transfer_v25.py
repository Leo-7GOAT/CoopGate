"""Descriptive event transport across every completed matched-transfer cell.

Read only the fully reviewed V21 transfer study. No active calibration records,
new simulations, refitting, selection or population certification are involved.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from coopgate.selective_transfer_v25 import empirical_coupling
from present_completed_v21 import load_complete
from profile_phase_runtime_v24 import digest,save
from study_matched_transfer_v21 import validate_grid
from study_temporal_v20 import is_failure

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/v21/interrupted_recovery_20261022/results/transfer'
OUT=ROOT/'artifacts/v25/matched_event_transfer'
FAMILIES=('raw','nominal','portfolio0','repair','phase05','phase10')
SEEDS=('1802','1812','1822')


def outcome(row,baseline):
    # Same original harmful-outcome definition, including every execution failure.
    harmful=bool(is_failure(row)
        or np.any(np.asarray(row['per_vehicle_cost'])-baseline['per_vehicle_cost']>1e-9)
        or np.any(np.asarray(row['per_vehicle_finish'])-baseline['per_vehicle_finish']>1e-9))
    adopted=row['learned_action_steps']>0
    return 2 if adopted and harmful else 1 if adopted else 0


def main():
    summaries,protocols,_=load_complete()
    spec=protocols['transfer']['spec']
    rows=json.loads((OLD/'episodes.json').read_text())
    if digest(OLD/'episodes.json')!=summaries['transfer']['episodes_sha256']:
        raise ValueError('Completed matched study changed')
    lookup=validate_grid(rows,spec)
    n=spec['n_initials_per_scenario']
    cells=[]
    for scenario in spec['scenarios']:
        for plant in spec['plants']:
            if plant=='nominal':continue
            for family in FAMILIES:
                for seed in SEEDS:
                    name=f'{family}_{seed}'
                    source=[lookup[scenario,i,'nominal',name] for i in range(n)]
                    target=[lookup[scenario,i,plant,name] for i in range(n)]
                    source_b=[lookup[scenario,i,'nominal','baseline'] for i in range(n)]
                    target_b=[lookup[scenario,i,plant,'baseline'] for i in range(n)]
                    for a,b,ba,bb in zip(source,target,source_b,target_b,strict=True):
                        np.testing.assert_array_equal(a['initial'],b['initial'])
                        if len({r['innovation_tape_sha256'] for r in (a,b,ba,bb)})!=1:
                            raise ValueError('Matched external noise required')
                    original=[outcome(a,b) for a,b in zip(source,source_b)]
                    changed=[outcome(a,b) for a,b in zip(target,target_b)]
                    diagnostic=empirical_coupling(original,changed)
                    cells.append(dict(scenario=scenario,source_plant='nominal',target_plant=plant,
                        family=family,model=seed,**diagnostic,
                        source_failures=sum(bool(is_failure(r)) for r in source),
                        target_failures=sum(bool(is_failure(r)) for r in target),
                        source_baseline_failures=sum(bool(is_failure(r)) for r in source_b),
                        target_baseline_failures=sum(bool(is_failure(r)) for r in target_b),
                        source_error_exceeded=sum(r['error_contract_exceeded_substeps']>0 for r in source),
                        target_error_exceeded=sum(r['error_contract_exceeded_substeps']>0 for r in target)))
    if len(cells)!=270:
        raise ValueError('All 5 scenes, 3 plant contrasts, 6 families, 3 models required')
    OUT.mkdir(exist_ok=False)
    report=dict(source_episodes_sha256=digest(OLD/'episodes.json'),source_summary_sha256=digest(OLD/'summary.json'),
        code_sha256={p:digest(ROOT/p) for p in ('src/coopgate/selective_transfer_v25.py','tools/audit_event_transfer_v25.py')},
        source_main_runs=12160,new_sumo_runs=0,new_independent_initials=0,
        independent_initials_per_scene=n,cells=cells,
        labels=['no adoption (including failures without adoption)','adopted without defined harm','adopted with defined harm'],
        scope='Post-hoc exact empirical event tables from completed development; no confidence or unknown population shift bound; no pooled IID scene claim',
        active_calibration_outcomes_used=False,risk_certified=False)
    save(OUT/'report.json',report)
    lines=['# 匹配动力学下的采用/退化事件变化','',
        '全部270个单元保留。每行32个旧初态；动力学、模型和方法不是新独立样本。下列风险为经验条件比例，'
        'TV和配对事件变化率也只描述这批样本，不提供未知部署总体的偏移上界或迁移风险证书。','',
        '源动力学均为nominal。目标动力学使用相同初态、外部噪声，但成本均与其各自动力学下的匹配基线比较。'
        '三状态为未采用、采用且无定义退化、采用且有定义退化；未采用时的失败另列，不会被状态压缩隐藏。','',
        '| 场景 | 目标动力学 | 方法/模型 | 源采用/有害 | 目标采用/有害 | 事件变化率 | 经验TV | 经验上界 | 目标失败 | 基线失败 | 误差越界 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in cells:
        a=r['source_counts'];b=r['target_counts'];bound=r['empirical_tv_envelope']
        text='未定义' if bound is None or not sum(b[1:]) else f"{100*bound['conditional_risk_upper']:.2f}%"
        lines.append(f"| {r['scenario']} | {r['target_plant']} | {r['family']}/{r['model']} | "
            f"{sum(a[1:])}/{a[2]} | {sum(b[1:])}/{b[2]} | {100*r['coupled_event_mismatch']:.2f}% | "
            f"{100*r['event_total_variation']:.2f}% | {text} | {r['target_failures']} | "
            f"{r['target_baseline_failures']} | {r['target_error_exceeded']} |")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'sha256.json',{p.name:digest(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(dict(cells=len(cells),new_sumo_runs=0,new_independent_initials=0,risk_certified=False)),flush=True)


if __name__=='__main__':main()
