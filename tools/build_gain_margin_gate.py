"""Generate a separate additional confidence condition without changing ranking."""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
from coopgate.scenario_confidence_gate import ScenarioConfidenceGate

ROOT=Path(__file__).resolve().parents[1]


def build():
    source=textwrap.dedent(inspect.getsource(ScenarioConfidenceGate._select));fn=ast.parse(source).body[0]
    start=ast.parse('self.gain_margin_evidence=[]\nself.selected_gain_margin=None\nself.selected_baseline_team_costs=[]').body
    fn.body[0:0]=start
    matches=[n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='passed']
    if len(matches)!=1:raise ValueError('Original successful candidate branch changed')
    extra=ast.parse('''
baseline_costs=[float(baseline_values[p][0][:,0].mean()) for p in range(path_count)]
score=relative_gain_score(differences,baseline_costs)
floor_passed=self.gain_floor==0 or (score is not None and score>=self.gain_floor)
self.gain_margin_evidence.append(dict(candidate=index,composition=code,repair_weight=candidates[index][1],
    baseline_team_costs=baseline_costs,worst_relative_team_gain=score,floor=self.gain_floor,passed=floor_passed))
if not floor_passed:
    candidate_evidence[-1].update(passed=False,gain_floor_rejected=True)
    continue
self.selected_gain_margin=score
self.selected_baseline_team_costs=baseline_costs
''').body
    # Preserve the original full-path completeness assertion before our test.
    matches[0].body[1:1]=extra
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict'
           and any(k.arg=='actual_learning_used' for k in n.keywords)]
    if len(calls)!=1:raise ValueError('Original final evidence mapping changed')
    extras=dict(performance_confidence_kind="'normalized_worst_path_team_gain'",gain_confidence_floor='self.gain_floor',
        selected_gain_confidence_score='self.selected_gain_margin',selected_baseline_forecast_team_costs='self.selected_baseline_team_costs',
        gain_margin_evidence='self.gain_margin_evidence',gain_score_is_probability='False')
    calls[0].keywords.extend(ast.keyword(arg=k,value=ast.parse(v,mode='eval').body) for k,v in extras.items())
    cls=ast.parse('''
class ScalarGainMarginGate(ScalarNominalParameterGate):
    def __init__(self,model,profile,gain_floor):
        if gain_floor not in (0.,.0025,.005,.01):raise ValueError('Use an explicitly registered gain floor')
        self.gain_floor=float(gain_floor)
        super().__init__(model,profile,0)
''').body[0]
    cls.body.append(fn)
    header='''"""Additional nondegenerate gain confidence; finite forecasts are NOT probabilities.

Generated from the original selector with ONLY one extra admission condition.
MAPPO actors,nominal ranking,per-car tests and physical certificates unchanged.
"""
import time
import numpy as np
from .scalar_nominal_recovery import ScalarNominalParameterGate
from .actuator_recovery_v34 import validate_inputs,stopping_certificate
from .ego_v13 import ego_observations
from .marl_v12 import LEVELS,action_mask
from .physical_switch_v27 import recovery_search
from .recovery_joint_gate_v35 import actuator_certificate_flags,compositions
from .temporal_portfolio_v20 import BANK_STREAM


def relative_gain_score(differences,baseline_team_costs):
    delta=np.asarray(differences,dtype=float);base=np.asarray(baseline_team_costs,dtype=float)
    if delta.shape!=(len(base),3,2) or len(base)==0 or not np.isfinite(delta).all() or not np.isfinite(base).all() or np.any(base<=0):
        return None
    return float(np.min(-delta[:,:,0].mean(axis=1)/base))


'''
    text='# original _select sha256='+hashlib.sha256(source.encode()).hexdigest()+'\n'+header+ast.unparse(ast.fix_missing_locations(cls))+'\n'
    target=ROOT/'src/coopgate/gain_margin_gate.py'
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    print(str(target),flush=True)


if __name__=='__main__':build()
