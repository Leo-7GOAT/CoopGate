"""Episode admission for both ego cost improvement and arrival non-degradation."""
import time
import numpy as np
from .ego_v13 import EgoRolloutGate
from .learned_rollout_v13 import LearnedRolloutGate,augmented_rollout
from .ego_v13 import ego_rollout
from .fast_v8 import advance_one,parameters
from .merge_v4 import MergeConfig
from .confidence_v12 import conflict_group,in_training_domain,conformal_upper


def predict_ego(initial,learned):
    c=MergeConfig();packed=parameters(c);results=[]
    for mode in ('baseline','learned'):
        x=np.array(initial,dtype=float);prev=np.zeros(3);elapsed=variation=0.
        for step in range(50):
            if x[1,0]>=c.goal:break
            if mode=='baseline':command,_,_,_=ego_rollout(x,prev,step,packed,.5,.05)
            else:command,_,_=augmented_rollout(x,prev,step,packed,.5,.05,learned.weights,learned.markov,learned.kind)
            elapsed+=.5;variation+=abs(command[1]-prev[1])
            if not advance_one(x,command,packed):raise RuntimeError('Invalid nominal forecast')
            prev=command
        results.append(np.array([elapsed+.05*variation,elapsed]))
    return results[1]-results[0]


class EgoConfidenceGate:
    """Choose a feedback controller once; certify its commands every cycle.

    q is a quantile of the maximum of two residuals, both expressed in seconds.
    It is neither a collision probability nor a per-decision safety score.
    """
    def __init__(self,policy_path,initial,calibration):
        start=time.perf_counter();self.baseline=EgoRolloutGate();self.learned=None;self.step=0
        self.accept=False;self.reason='outside_calibration_domain';self.predicted=np.array([np.nan,np.nan])
        self.upper=np.array([np.inf,np.inf]);self.fault_fallback=False
        if in_training_domain(np.asarray(initial)):
            try:
                self.learned=LearnedRolloutGate(policy_path);self.predicted=predict_ego(initial,self.learned)
                entry=calibration['groups'][conflict_group(initial)];q=float(entry['q'])
                self.upper=self.predicted+q
                self.accept=bool(self.upper[0]<=-.05 and self.upper[1]<=0.)
                self.reason='admitted' if self.accept else ('arrival_uncertain' if self.upper[1]>0 else 'insufficient_cost_gain')
            except (RuntimeError,ValueError,KeyError,OSError):self.reason='prediction_failed'
        self.admission_ms=(time.perf_counter()-start)*1000

    def select(self,x,previous,unused=None):
        start=time.perf_counter()
        if self.accept and not self.fault_fallback:
            try:
                self.learned.step=self.step;command,info=self.learned.select(x,previous)
                if not np.all(np.isfinite(command)):raise ValueError('Nonfinite learned command')
            except (RuntimeError,ValueError,TypeError):self.fault_fallback=True
        if not self.accept or self.fault_fallback:
            self.baseline.step=self.step;command,info=self.baseline.select(x,previous)
        self.step+=1;info.update(performance_admitted=self.accept,fault_fallback=self.fault_fallback,
            gate_ms=(time.perf_counter()-start)*1000)
        return command,info
