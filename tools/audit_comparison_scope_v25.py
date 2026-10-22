"""Report every fixed controller under the SAME previously computed LTT rule.

This does not change thresholds, select a model, recertify an old study, or read
unfinished calibration. The prospective entry requires both complete stages.
Historical V19 data exercise the reporting path without opening V21 outcomes.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

from probe_disturbed_gate_v19 import digest
from recover_interrupted_v21 import manifest, read
from start_final_v19 import require_committed, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('tools/audit_comparison_scope_v25.py','tests/test_comparison_scope_v25.py')
MODELS = ('1802','1812','1822')


def checked_ltt(value, names):
    if (value['procedure'] != 'bonferroni' or value['order'] is not None
            or type(value['independent_initials']) is not int or value['independent_initials'] <= 0
            or not 0 < value['alpha'] < 1 or not 0 < value['delta'] < 1):
        raise ValueError('Uniform report requires the original common Bonferroni rule')
    candidates = value['candidates']
    if len(candidates) != len(names) or {r['candidate'] for r in candidates} != set(names):
        raise ValueError('All nine fixed candidates must appear exactly once')
    level = value['delta']/len(names)
    by_name = {}
    for row in candidates:
        n,k = row['adopted'],row['harmful']
        if (any(type(v) is not int for v in (n,k)) or not 0 <= k <= n <= value['independent_initials']
                or row['tested'] is not True or type(row['certified']) is not bool):
            raise ValueError('Invalid adoption-subset evidence')
        p = float(binom.cdf(k,n,value['alpha'])) if n else 1.
        upper = float(beta.ppf(1-level,k+1,n-k)) if k<n else 1.
        empirical = k/n if n else None
        if (row['certified'] != bool(n>0 and p<=level)
                or row['empirical_conditional_risk'] != empirical
                or not np.allclose([row['binomial_pvalue'],row['threshold_upper']],[p,upper],rtol=1e-12,atol=1e-14)):
            raise ValueError('Stored LTT result does not match its original counts and level')
        by_name[row['candidate']] = row
    return by_name


def shared_table(actual, bundle, results, family):
    prefixes = ('raw','nominal',family)
    if len(set(prefixes)) != 3:
        raise ValueError('Three distinct complete controller families required')
    names = tuple(f'{prefix}_{seed}' for prefix in prefixes for seed in MODELS)
    for key in ('independent_initials','alpha','delta','procedure'):
        if actual[key] != bundle[key]:
            raise ValueError('Both targets must have the same declared rule and budget')
    a,b = checked_ltt(actual,names),checked_ltt(bundle,names)
    if 2*actual['delta']>=1:
        raise ValueError('Nonvacuous simultaneous comparison budget required')
    rows = []
    final_n = set()
    for prefix in prefixes:
        members = results[prefix]['per_training_seed']
        if set(members) != set(MODELS):
            raise ValueError('Do not drop or choose training models')
        for seed in MODELS:
            name = f'{prefix}_{seed}'
            value = members[seed]
            final_n.add(value['runs'])
            if value['selected_profiles'] != {name:value['runs']}:
                raise ValueError('A fixed controller row cannot contain a selected or mixed profile')
            final = value['actual_random_mode']
            n,k = final['adopted'],final['harmful_adopted']
            if (not 0 <= k <= n <= value['runs'] or value['runs']<=0
                    or final['conditional_harm'] != (k/n if n else None)
                    or final['adoption_fraction'] != n/value['runs']):
                raise ValueError('Final evidence must preserve the actual adoption denominator')
            rows.append(dict(controller=prefix,model=seed,calibration_actual=a[name],calibration_bundle=b[name],
                same_rule_both_targets_certified=bool(a[name]['certified'] and b[name]['certified']),
                final_runs=value['runs'],final_cost_gain_percent=value['mean_team_gain_percent'],
                final_cost_gain_seconds=value['mean_team_gain_seconds'],final_gain_ci95=value['mean_team_gain_ci95'],
                final_actual_adoption=final,final_bundle=value['worst_mode_bundle'],
                final_completed=value['completed'],final_controller_p95_ms=value['controller_p95_ms']))
    if len(final_n) != 1:
        raise ValueError('Every fixed controller requires the same complete final population')
    return dict(calibration_initials=actual['independent_initials'],final_initials=final_n.pop(),
        risk_limit=actual['alpha'],per_target_family_error=actual['delta'],
        per_candidate_target_test_level=actual['delta']/9,
        simultaneous_error_budget_both_targets=2*actual['delta'],
        rows=rows,whole_groups_under_shared_rule={p:all(r['same_rule_both_targets_certified'] for r in rows if r['controller']==p) for p in prefixes},
        inference_rule_identical_for_all_nine=True,old_certificates_recomputed_at_new_levels=False,
        cert_difference_is_not_itself_gate_improvement=True,
        statewise_confidence=False,finite_population_sampling_unit='independent initial, never a mode or controller replay')


def historical_inputs():
    cal_dir = ROOT/'artifacts/v19/comparison_calibration'
    final_dir = ROOT/'artifacts/v19/final_comparison'
    verify_manifest(cal_dir)
    verify_manifest(final_dir)
    cal,final = read(cal_dir/'calibration.json'),read(final_dir/'comparison.json')
    risk = read(cal_dir/'conditional_risk.json')
    if (cal['conditional_risk_sha256'] != digest(cal_dir/'conditional_risk.json')
            or final['calibration_sha256'] != digest(cal_dir/'calibration.json')
            or final['plan_sha256'] != cal['plan_sha256']):
        raise ValueError('Completed historical comparison provenance changed')
    paths = [cal_dir/'calibration.json',cal_dir/'conditional_risk.json',final_dir/'comparison.json']
    return (risk['targets']['actual_random_mode'],risk['targets']['worst_mode_bundle'],
            final['results'],'portfolio0',None,paths)


def prospective_inputs():
    import recover_prospective_v24 as recovery
    _,plan = recovery.inputs()
    # No unfinished calibration effects are read, even for this descriptive table.
    recovery.complete_stage(plan,'calibration')
    recovery.complete_stage(plan,'confirmation')
    cal_dir = recovery.OUTPUT/'calibration/analysis'
    final_dir = recovery.OUTPUT/'confirmation/analysis'
    verify_manifest(cal_dir)
    verify_manifest(final_dir)
    paths = [cal_dir/'calibration.json',final_dir/'comparison.json',final_dir/'acceptance.json']
    require_committed(paths)
    cal,final = read(paths[0]),read(paths[1])
    if (final['calibration_sha256'] != digest(paths[0])
            or cal['execution_plan_sha256'] != final['execution_plan_sha256']
            or cal['recovery_plan_sha256'] != final['recovery_plan_sha256']):
        raise ValueError('Complete prospective comparison provenance changed')
    return cal['ltt_actual'],cal['ltt_bundle'],final['results'],plan['family'],cal['primary_certificate'],paths


def markdown(report):
    lines = ['# 共同LTT规则下的完整控制器对照','',
        '本表只重现既有校准结果，并列展示独立最终测试；没有改变统计水平、重新认证或增加独立样本。','',
        '| 控制器 | 模型 | 校准实际风险上界 | 校准组合风险上界 | 同规则两项通过 | 最终成本改善 | 最终采用率 | 最终采用后退化 |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for row in report['rows']:
        a,b = row['calibration_actual'],row['calibration_bundle']
        f = row['final_actual_adoption']
        lines.append(f"| {row['controller']} | {row['model']} | {100*a['threshold_upper']:.4f}% | {100*b['threshold_upper']:.4f}% | "
            f"{'是' if row['same_rule_both_targets_certified'] else '否'} | {row['final_cost_gain_percent']:.4f}% | "
            f"{100*f['adoption_fraction']:.4f}% | {f['harmful_adopted']}/{f['adopted']} |")
    lines += ['',f"每个候选/目标的原定检验水平为 {report['per_candidate_target_test_level']:.12g}；"
        f"两个目标合计的原定多重检验错误预算为 {report['simultaneous_error_budget_both_targets']:.4f}。",
        '校准上界与最终经验退化比例来自不同数据集。零采用的条件风险未定义，不能记成零风险。',
        '主方法的完整组交并检验与此表的可选择候选证书是不同命题；不得把二者通过率差异归因于门控算法改进。','']
    return '\n'.join(lines)


def main(scope):
    require_committed([ROOT/p for p in SOURCES])
    actual,bundle,results,family,primary,paths = historical_inputs() if scope=='historical' else prospective_inputs()
    report = shared_table(actual,bundle,results,family)
    report.update(scope=scope,utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name:digest(ROOT/name) for name in SOURCES},
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        primary_all_or_none_certificate=primary,primary_test_replaced=False,
        additional_sumo_runs=0,additional_independent_initials=0,new_risk_certificate=False,
        unfinished_prospective_outcomes_read=False)
    directory=ROOT/'artifacts/v25'/f'{scope}_common_inference_audit'
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    (directory/'report.md').write_text(markdown(report),encoding='utf-8',newline='\n')
    (directory/'sha256.json').write_text(json.dumps(manifest(directory),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(scope=scope,rows=len(report['rows']),whole_groups_under_shared_rule=report['whole_groups_under_shared_rule'],
                         no_new_risk_claim=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('scope',choices=('historical','prospective'))
    main(parser.parse_args().scope)
