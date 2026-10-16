"""Refuse multi-agent training until the complete V14 prerequisite is audited."""
import hashlib,json
from pathlib import Path


def require_single(root):
    root=Path(root);data=root/'artifacts/v14/single'
    paths=[data/'protocol.json',data/'summary.json',data/'sha256.json',root/'artifacts/v14/single_audit.json']
    if not all(path.is_file() for path in paths):raise RuntimeError('V14 complete single-ego acceptance/replay evidence is required')
    protocol,summary,hashes,audit=[json.loads(path.read_text()) for path in paths]
    if not (summary['individual_ego_verified'] and summary['acceptance']['all_passed'] and audit['individual_ego_verified']):
        raise RuntimeError('Single-ego superiority prerequisite is unmet')
    if audit['acceptance']!=summary['acceptance'] or audit['compiled_actor_comparisons']<=0:
        raise RuntimeError('Acceptance or actor audit is inconsistent')
    expected=protocol['expected_runs'];modes=len(protocol['disturbances'])*len(protocol['methods'])
    if audit['validated_sumo_runs']!=expected or audit['sumo_replays']!=modes*len(protocol['replay_initial_indices']):
        raise RuntimeError('Incomplete formal study or SUMO replays')
    if audit['audited_traces']!=modes*len(protocol['audit_initial_indices']):raise RuntimeError('Incomplete trajectory audit')
    for name,digest in protocol['input_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Changed frozen input: {name}')
    for name,digest in hashes.items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Changed frozen result: {name}')
    if audit['protocol_sha256']!=hashlib.sha256(paths[0].read_bytes()).hexdigest():raise RuntimeError('Audit protocol mismatch')
    if audit['episodes_sha256']!=hashlib.sha256((data/'study/episodes.json').read_bytes()).hexdigest():raise RuntimeError('Audit execution mismatch')
    return dict(protocol_sha256=audit['protocol_sha256'],episodes_sha256=audit['episodes_sha256'],
                audit_sha256=hashlib.sha256(paths[-1].read_bytes()).hexdigest())
