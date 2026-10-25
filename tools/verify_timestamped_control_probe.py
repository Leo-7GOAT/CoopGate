"""Frozen whole-study replay after actual native exit, retaining all failures."""
import numpy as np

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import observations
import execute_timestamped_control_probe as run

io=run.io


def controller_scopes(paired):
    result=[]
    for variant in run.VARIANTS:
        records=[r for r in paired if r['variant']==variant]
        references={r['source_index']:r['row'] for r in records if r['row']['controller']=='baseline'}
        assert set(references)=={0,1,2}
        for controller in sorted({r['row']['controller'] for r in records}):
            selected=[r for r in records if r['row']['controller']==controller]
            assert len(selected)==3 and {r['source_index'] for r in selected}=={0,1,2}
            assessable=all(not r['row']['failure'] and not references[r['source_index']]['failure'] for r in selected)
            before=sum(references[r['source_index']]['team_cost'] for r in selected)
            after=sum(r['row']['team_cost'] for r in selected)
            result.append(dict(variant=variant,controller=controller,conditions=3,
                failed_conditions=sum(r['row']['failure'] for r in selected),
                baseline_failed_conditions=sum(references[r['source_index']]['failure'] for r in selected),
                all_scope_cost_gain_percent=100*(before-after)/before if assessable else None))
    return result


def verify():
    p=run.inputs();out=run.OUT
    done=io.read(out/'complete.json');exit_proof=io.read(out/'observed_execution_exit.json')
    assert exit_proof['exit_code']==0 and exit_proof['native_pid']==io.read(out/'native_start.json')['native_pid']
    assert exit_proof['complete_sha256']==io.digest(out/'complete.json')
    assert done['protocol_sha256']==io.digest(run.PLAN) and done['actual_sumo_runs']==117
    assert done['all_failures_retained'] and done['raw_non_timing_identity_verified']
    assert [{k:e[k] for k in ('index','variant','source_index')} for e in done['blocks']]==p['jobs']
    for name,sha in done['files_sha256'].items():assert io.digest(out/name)==sha
    original,_=run.source_case(0)
    models={name:load_coupled(run.ROOT/item['path'])[0] for name,item in original['models'].items()}
    paired=[];physical_steps=frames=terminal=0
    for job,entry in zip(p['jobs'],done['blocks']):
        plan,group=run.source_case(job['source_index']);variant=job['variant']
        directory=out/'blocks'/f"{job['index']:03d}"
        complete=io.read(directory/'complete.json')
        assert io.digest(directory/'complete.json')==entry['complete_sha256']
        meta=run.metadata(group,variant,io.digest(run.PLAN))
        assert all(complete[k]==v for k,v in meta.items()) and complete['actual_sumo_runs']==13
        assert complete['database_sha256']==io.digest(directory/'observations.sqlite')
        assert complete['seal_sha256']==io.digest(directory/'observations.sqlite.seal.json')
        _,auditor=run.native_functions(variant);reviews=[]
        records=list(observations(directory/'observations.sqlite',meta,io.schedule(group)))
        assert len(records)==13
        for key,body,trace in records:
            method=dict(rule=key[-2],model=key[-1])
            model=None if method['model']=='none' else models[method['model']]
            review=auditor(plan,group,method,body,trace,model)
            reviews.append(dict(case=list(key),**review))
            physical_steps+=review['physical_substeps'];frames+=review['received_frames']
            terminal+=review['terminal_error_replayed']
            paired.append(dict(job_index=job['index'],variant=variant,case=list(key),
                source_index=job['source_index'],row=body['row'],review=review))
        assert reviews==complete['reviews']
        if variant=='projected_raw':run.compare_raw(directory,group,io.digest(run.PLAN))
    assert len(paired)==117
    aggregate=[]
    for variant in run.VARIANTS:
        for source_index in range(3):
            selected=[r for r in paired if r['variant']==variant and r['source_index']==source_index]
            assert len(selected)==13
            base=next(r['row'] for r in selected if r['row']['rule']=='baseline')
            bank=[r['row'] for r in selected if r['row']['rule']=='bank_performance']
            assert len(bank)==3
            bank_gains={r['model']:(100*(base['team_cost']-r['team_cost'])/base['team_cost']
                if not base['failure'] and not r['failure'] else None) for r in bank}
            rows=[r['row'] for r in selected]
            aggregate.append(dict(variant=variant,source_index=source_index,seed=rows[0]['seed'],
                perception=rows[0]['perception_name'],episodes=13,
                tasks_completed=sum(r['completed'] for r in rows),failed_cases=sum(r['failure'] for r in rows),
                aborts=sum(r['aborted'] for r in rows),collisions=sum(r['collision'] for r in rows),
                geometry_violation_cases=sum(r['contract_violations']>0 for r in rows),
                adopted_cases=sum(r['adopted'] for r in rows),
                estimator_failure_frames=sum(r['review']['received_frames']-r['review']['envelope_frames']
                                             for r in selected),
                baseline_completed=base['completed'],baseline_failed=base['failure'],
                baseline_team_cost=base['team_cost'],bank_gains_vs_own_baseline_percent=bank_gains))
    assert run.inputs()==p
    return dict(complete_sha256=io.digest(out/'complete.json'),observed_exit=exit_proof,
        actual_sumo_runs=117,complete_blocks=9,physical_substeps=physical_steps,received_frames=frames,
        terminal_errors_replayed=terminal,all_physics_perception_and_preprocessing_replayed=True,
        raw_39_controls_match_prior_v45_non_timing=True,aggregate=aggregate,
        controller_scopes=controller_scopes(paired),paired=paired,
        reused_independent_initials=3,new_independent_initials=0,
        formal_studies_unchanged=True,partial_formal_efficacy_read=False,
        posthoc_development_only=True,physical_certificate_restored=False,new_risk_certificate=False,
        interval_midpoint_is_not_exact_state=True,marl_training_claimed=False)
