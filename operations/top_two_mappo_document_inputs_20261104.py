"""Pure JSON/SHA checks for the bundled document runtime; no simulator imports."""


def load(bundle,read,sha,ROOT,FORMAL,BASE,INPUTS):
    terminal=read(bundle.with_suffix('.native.json'))
    if terminal['actual_native_exit_code']!=0 or terminal['bundle_sha256']!=sha(bundle):
        raise ValueError('Actual successful complete-data exporter required before authoring')
    data=read(bundle)
    if not data['completed'] or data['source_sha256']!=sha(ROOT/'operations/build_top_two_paper_evidence_20261104.py'):
        raise ValueError('Exact complete new exporter source required')
    owner=read(FORMAL/'orchestrator_native_receipt.json');receipt=read(FORMAL/'completion_receipt.json')
    if (owner['actual_native_exit_code']!=0 or not owner['observed_by_direct_native_wait']
        or owner['original_owner_exit_code'] is not None
        or owner['completion_receipt_sha256']!=sha(FORMAL/'completion_receipt.json')
        or receipt['actual_native_finalizer']['actual_native_exit_code']!=0
        or receipt['total_actual_sumo_runs']!=59392 or receipt['independent_initials']!=7168
        or len(receipt['stages'])!=16 or any(q['actual_native_exit_code']!=0 for q in receipt['stages'])):
        raise ValueError('Actual complete recovery owner and every fixed stage required')
    accounting=read(FORMAL/'recovery_delivery_accounting.json')
    if (not accounting['complete'] or accounting['retained_actual_records']!=642
        or accounting['new_actual_sumo_executions']!=58750 or accounting['original_root_exit_code'] is not None
        or accounting['new_error_allocated']):
        raise ValueError('Keep interruption and original scope explicit')
    cert=read(FORMAL/'analysis/certificate.json');comparison=read(FORMAL/'analysis/comparisons.json')
    if (cert!=data['full_primary_certificate'] or comparison!=data['all_confirmation_comparisons']
        or sha(FORMAL/'analysis/certificate.json')!=data['certificate_sha256']
        or sha(FORMAL/'analysis/comparisons.json')!=data['comparisons_sha256']):
        raise ValueError('Complete evidence differs from formal inference')
    if (cert['attempt']!=6 or cert['allocated_error']!=.05/42 or cert['fixed_models']!=[4703,4704,4705]
        or cert['common_gain_floor']!=.0025 or cert['calibration_initials']!=6144 or cert['confirmation_initials']!=1024
        or cert['primary_risk']['risk_limit']!=.05 or cert['adoption']['threshold']!=.2
        or cert['paired_gain']['threshold_percent']!=.5):
        raise ValueError('Original scientific requirements changed')
    plan=read(FORMAL/'plan.json')
    def bind(name,digest):
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path)!=digest:
            raise ValueError('Completed input changed: '+name)
        INPUTS[name]=digest
    for name,digest in plan['source_sha256'].items():bind(name,digest)
    for name,digest in data['postprocessing_sources_sha256'].items():bind(name,digest)
    recovery=read(FORMAL/'recovery_binding.json')
    bind('tools/recover_mappo_top_two_attempt6.py',recovery['source_sha256'])
    bind('docs/mappo_attempt6_interruption_recovery_20261104.md',recovery['protocol_sha256'])
    for item in data['complete_raw_collection_manifest']:
        directory=FORMAL/item['stage']/('shard'+str(item['shard']))
        path=directory/'collection_complete.json'
        if sha(path)!=item['collection_complete_sha256']:raise ValueError('Complete raw manifest changed')
        complete=read(path)
        for block in complete['blocks']:
            for name,key in [('observations.sqlite','database_sha256'),('observations.sqlite.seal.json','seal_sha256')]:
                path=directory/block['block']/name
                bind(path.relative_to(ROOT).as_posix(),block[key])
    legacy_path=BASE/'powered_attempt5_manuscript_evidence_20261103.json'
    legacy_terminal=read(legacy_path.with_suffix('.native.json'));legacy=read(legacy_path)
    if (legacy_terminal['actual_native_exit_code']!=0 or legacy_terminal['bundle_sha256']!=sha(legacy_path)
        or legacy['exporter_sha256']!=sha(ROOT/'operations/build_powered_history_evidence_20261103.py')):
        raise ValueError('Retained physical/training/theory evidence is not terminal-bound')
    for name,digest in legacy['inputs_sha256'].items():bind(name,digest)
    for name in ('operations/top_two_mappo_document_inputs_20261104.py',
                 'operations/top_two_mappo_manuscript_text_20261104.py',
                 'operations/powered_mappo_manuscript_text_20261103.py',
                 'operations/build_completed_manuscript_20261101.py'):
        bind(name,sha(ROOT/name))
    return cert,comparison,dict(new=data,old=legacy)
