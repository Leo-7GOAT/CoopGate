"""Unchanged full V42 study after explicitly authorized obsolete-attempt withdrawal."""
import argparse
from types import FunctionType
import execute_joint_risk_v42 as original
import legacy_withdrawal_contract_20261027 as amendment
import finalize_episode_keys as keys
import parallel_joint_risk_replay_20261027 as replay

SCRIPT=amendment.ROUTE


def routed_functions():
    namespace={**vars(original),'SCRIPT':SCRIPT}
    for name,function in vars(original).items():
        if isinstance(function,FunctionType) and function.__module__==original.__name__:
            clone=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
            clone.__kwdefaults__=function.__kwdefaults__;namespace[name]=clone
    namespace['verify_previous_attempt']=amendment.prerequisite
    return namespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('bind','finalize'):commands.add_parser(name)
    for name in ('watch','worker'):commands.add_parser(name).add_argument('stage',choices=original.design.STAGES)
    args=parser.parse_args()
    amendment.inputs(full=args.command in ('bind','finalize'))
    namespace=routed_functions()
    if args.command=='bind':
        replay.contract_inputs();namespace['bind']();return replay.bind_registration()
    replay.bound_registration_inputs()
    if args.command=='finalize':
        keys.inputs();replay.finalize(namespace)
        target=original.OUT/'analysis'
        original.io.exclusive_json(target/'authorized_prerequisite_amendment.json',dict(
            amendment_sha256=amendment.digest(amendment.PLAN),
            legacy_status='withdrawn incomplete; no certificate',
            legacy_error_budget_reassigned=False,full_required_new_runs=16384,
            original_scientific_function_bodies_unchanged=True,
            original_statistical_contract_sha256=amendment.canonical(original.io.read(original.PLAN)['contract'])))
        return
    namespace[args.command](args.stage)


if __name__=='__main__':main()
