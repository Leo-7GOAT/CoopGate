"""Generate an explicit score-threshold gate; no frozen source is modified."""
import ast
import inspect
from pathlib import Path
import textwrap

from coopgate.recovery_joint_gate_v35 import RecoveryJointGate

ROOT=Path(__file__).resolve().parents[1]


def build():
    tree=ast.parse(textwrap.dedent(inspect.getsource(RecoveryJointGate.select)));fn=tree.body[0];fn.name='_select'
    changed=0
    for node in ast.walk(fn):
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='path':
            node.body.insert(0,ast.parse('if path == 0: violating_paths = 0').body[0])
            for statement in node.body:
                if isinstance(statement,ast.If) and 'np.any(delta' in ast.unparse(statement.test):
                    statement.body=ast.parse('violating_paths += 1\nif violating_paths > self.max_failed_paths:\n    passed = False\n    break').body
                    changed+=1
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
            names={k.arg for k in node.keywords}
            if {'candidate','composition','passed','checked_paths'}<=names:
                node.keywords.append(ast.keyword(arg='violating_paths',value=ast.Name(id='violating_paths',ctx=ast.Load())))
            if 'conditional_performance_risk_certified' in names:
                score=ast.parse('(path_count-violating_paths)/path_count if used else None',mode='eval').body
                node.keywords.append(ast.keyword(arg='scenario_pass_fraction',value=score))
    if changed!=1:raise ValueError('Original gate shape changed; review generation explicitly')
    body=textwrap.indent(ast.unparse(ast.fix_missing_locations(fn)),'    ')
    header='''"""Fixed-scenario confidence score with separate trajectory-level risk validation.

Generated from the original selector by tools/build_scenario_confidence_gate.py.
The score is a deterministic scenario pass fraction, not a calibrated safety
probability. Physical admission remains the same actuator stopping certificate.
"""
import time
import numpy as np
from .actuator_recovery_v34 import validate_inputs,stopping_certificate
from .ego_v13 import ego_observations
from .marl_v12 import LEVELS,action_mask
from .physical_switch_v27 import recovery_search
from .recovery_joint_gate_v35 import actuator_certificate_flags,compositions
from .memoized_recovery_v37 import MemoizedRecoveryGate,CACHE_FIELDS,CAPACITY
from .temporal_portfolio_v20 import BANK_STREAM

class ScenarioConfidenceGate(MemoizedRecoveryGate):
    def __init__(self,model,profile,max_failed_paths=0):
        if type(max_failed_paths) is not int or max_failed_paths not in (0,1,3):
            raise ValueError('Use an explicit registered scenario budget0,1 or3')
        super().__init__(model,profile,'bank_performance')
        self.max_failed_paths=max_failed_paths

    def select(self,state,previous,unused=None):
        started=time.perf_counter()
        command,info=self._select(state,previous,unused)
        current=getattr(self,'_cache_step',None)==self.step-1
        values=self._cache_stats.tolist() if current else [0]*len(CACHE_FIELDS)
        if values[0]!=values[1]+values[2] or values[5]!=info['completed_planning_model_transitions']:
            raise RuntimeError('Scenario gate planning work accounting mismatch')
        info.update(rule='scenario_confidence',scenario_max_failed_paths=self.max_failed_paths,
            confidence_threshold=(len(self.tapes)-self.max_failed_paths)/len(self.tapes),
            scenario_score_is_probability=False,conditional_performance_risk_certified=False,
            proposal_cache=dict(zip(CACHE_FIELDS,map(int,values))),
            proposal_cache_entries=len(self._plan_cache) if current else 0,proposal_cache_capacity=CAPACITY,
            planning_counter_scope='Logical reference planning and separately counted actual cache work; no shared selector budget.',
            gate_ms=1000*(time.perf_counter()-started))
        if info['actual_learning_used'] and info['scenario_pass_fraction']+1e-12<info['confidence_threshold']:
            raise RuntimeError('Executed learning below the declared confidence-score threshold')
        return command,info

'''
    target=ROOT/'src/coopgate/scenario_confidence_gate.py'
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(header+body+'\n')
    ast.parse(target.read_text());print(str(target),flush=True)


if __name__=='__main__':build()
