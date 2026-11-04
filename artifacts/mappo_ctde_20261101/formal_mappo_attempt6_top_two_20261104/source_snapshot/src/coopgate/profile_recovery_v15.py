"""Physical runtime recovery around an externally selected joint policy profile.

This wrapper does not calculate or certify statistical confidence. A separately
frozen/calibrated selector supplies the admission decision. Unmodelled runtime
faults do not inherit a performance-risk bound from fault-free calibration.
"""
import time
import numpy as np
from .merge_v4 import MergeConfig
from .fast_v8 import parameters
from .robust_v9 import validate,robust_certificate


class RecoveringProfile:
    def __init__(self,primary,baseline,admitted,config=None,predicted=None,upper=None,admission_ms=0.):
        self.primary=primary;self.baseline=baseline;self.c=config or MergeConfig();self.packed=parameters(self.c)
        self.step=0;self.accept=bool(admitted);self.fault_fallback=False
        if self.accept and primary is None:raise ValueError('An admitted controller is required')
        self.reason='admitted' if self.accept else 'not_admitted';self.admission_ms=float(admission_ms)
        self.predicted=np.full((3,2),np.nan) if predicted is None else np.array(predicted,dtype=float)
        self.upper=np.full((3,2),np.inf) if upper is None else np.array(upper,dtype=float)
        self.fault_reason=''

    def checked(self,controller,x,previous):
        controller.step=self.step
        # Preserve the measured state/last actual command even if a failed
        # controller accidentally mutates its inputs before raising an error.
        command,info=controller.select(x.copy(),previous.copy())
        command=np.asarray(command,dtype=float)
        if command.shape!=(3,) or not np.isfinite(command).all():raise ValueError('Nonfinite or malformed command')
        if np.any(command < -self.c.brake-1e-9) or np.any(command > self.c.accel+1e-9):raise ValueError('Acceleration limit')
        if np.any(np.abs(command-previous)>self.c.jerk*self.c.decision_dt+1e-9):raise ValueError('Command-slew limit')
        if not robust_certificate(x,previous,command,self.packed,.5):raise ValueError('Missing physical certificate')
        return command,dict(info)

    def select(self,state,previous,unused=None):
        start=time.perf_counter();x,prev,_=validate(state,previous,self.c,.5)
        backup=np.maximum(prev-self.c.jerk*self.c.decision_dt,-self.c.brake)
        if not robust_certificate(x,prev,backup,self.packed,.5):raise RuntimeError('Outside recoverable contract')
        result=None
        if self.accept and not self.fault_fallback:
            try:
                result=self.checked(self.primary,x,prev)
            except (RuntimeError,ValueError,TypeError,IndexError,KeyError,OSError) as exc:
                self.fault_fallback=True;self.fault_reason=f'{type(exc).__name__}: {exc}'
                self.reason='runtime_fault_fallback'
        if result is None:result=self.checked(self.baseline,x,prev)
        command,info=result;self.step+=1
        # Initial admission is retained after a fault: failures cannot disappear
        # from the denominator of admitted-episode performance risk.
        info.update(in_set=True,certified=True,performance_admitted=self.accept,
            fault_fallback=self.fault_fallback,fault_reason=self.fault_reason,
            gate_ms=(time.perf_counter()-start)*1000.)
        return command,info
