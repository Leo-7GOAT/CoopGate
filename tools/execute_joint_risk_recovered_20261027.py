"""Original V42 execution, routed to the unchanged fifth recovery of attempt 1.

No binding or formal collection until BOTH predecessor stages fully complete.
Scientific function bodies, output plan, models, budgets and statistics remain
the original ones. The existing frozen tuple/list adapter is used at finalize.
"""
import argparse
from types import FunctionType

import execute_joint_risk_v42 as original
import finalize_episode_keys as keys
import recover_prospective_20261027 as recovered

SCRIPT='tools/execute_joint_risk_recovered_20261027.py'


def routed_functions():
    namespace={**vars(original),'predecessor':recovered,'SCRIPT':SCRIPT}
    for name,function in vars(original).items():
        if isinstance(function,FunctionType) and function.__module__==original.__name__:
            clone=FunctionType(function.__code__,namespace,function.__name__,
                               function.__defaults__,function.__closure__)
            clone.__kwdefaults__=function.__kwdefaults__
            namespace[name]=clone
    return namespace


def finalize(namespace):
    keys.inputs()
    import parallel_joint_risk_replay_20261027 as replay
    replay.finalize(namespace)
    target=original.OUT/'analysis'
    outputs=[target/name for name in ('certificate.json','replay.json','verification.json')]
    original.io.exclusive_json(target/'key_representation_compatibility.json',dict(
        compatibility_plan_sha256=keys.files.digest(keys.PLAN),target='joint-risk',
        target_source_sha256=original.io.digest(original.ROOT/keys.TARGET_FILES['joint-risk']),
        frozen_function_body_unchanged=True,private_key_container_adapter_used=True,
        module_globals_mutated=False,
        output_sha256={p.relative_to(original.ROOT).as_posix():original.io.digest(p) for p in outputs},
        predecessor_output=recovered.OUTPUT.relative_to(original.ROOT).as_posix(),
        execution_route_source_sha256=original.io.digest(original.ROOT/SCRIPT)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('bind','finalize'):commands.add_parser(name)
    for name in ('watch','worker'):
        commands.add_parser(name).add_argument('stage',choices=original.design.STAGES)
    args=parser.parse_args(); namespace=routed_functions()
    import parallel_joint_risk_replay_20261027 as replay
    # Original bind freezes every tools/*.py source, including this route.
    # Actual terminal receipts, full replay and both original attempts' alpha
    # accounting are checked by the unchanged predecessor verification body.
    if args.command=='finalize':return finalize(namespace)
    if args.command=='bind':
        replay.contract_inputs()
        namespace['bind']()
        return replay.bind_registration()
    replay.bound_registration_inputs()
    namespace[args.command](*([args.stage] if hasattr(args,'stage') else []))


if __name__=='__main__':main()
