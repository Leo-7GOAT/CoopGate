"""Finite forecast witness fitted ONLY to measured boundary history.

Unit gain, zero drag and five semi-implicit .1s substeps are explicit model
assumptions. A fitted constant innovation is a hypothesis about the future,
not a posterior, disturbance guarantee, or performance certificate.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from .actuator_recovery_v34 import class_bounds


def fit_observed_history(states, commands, profile):
    """Return one bounded lag/bias witness or None without sufficient history.

    states[k] precedes executed commands[k]. There must be one more measured
    state than commands; actuator memory at the experiment reset is zero.
    Windows near the speed clip are excluded from the linear bias formula.
    Neither realized parameters nor future innovations are accepted as inputs.
    """
    if profile not in ('nominal', 'lag_mild', 'lag_strong'):
        raise ValueError('Only declared unit-gain zero-drag classes supported')
    x=np.asarray(states,dtype=float);u=np.asarray(commands,dtype=float)
    if x.ndim!=3 or x.shape[1:]!=(3,2) or u.shape!=(len(x)-1,3):
        raise ValueError('Measured N+1 states and N executed commands required')
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError('Finite observations required')
    if np.any(x[:,:,1]<0) or np.any(x[:,:,1]>20) or np.any(u<-5) or np.any(u>2.5):
        raise ValueError('Outside declared speed/command bounds')
    if not len(u):return None
    bounds=class_bounds(profile);rhos=[];biases=[];memories=[];errors=[];usable=[]
    for car in range(3):
        # |m+b| <=5.1 guarantees no speed clipping within a .5s window
        # for these measured starting speeds, independently of the fitted rho.
        keep=(x[:-1,car,1]>2.55)&(x[:-1,car,1]<17.45)
        if not keep.any():return None
        def evaluate(rho,details=False):
            memory=0.;dv=[];dp=[]
            for k,command in enumerate(u[:,car]):
                total=weighted=0.
                for s in range(5):
                    memory=rho*memory+(1-rho)*command
                    total+=memory;weighted+=(5-s)*memory
                dv.append(x[k+1,car,1]-x[k,car,1]-.1*total)
                dp.append((x[k+1,car,0]-x[k,car,0]-.5*x[k,car,1]-.01*weighted)/.5)
            rv=np.asarray(dv)[keep];rp=np.asarray(dp)[keep]
            bias=float(np.clip((.5*rv.sum()+.3*rp.sum())/(.34*keep.sum()),-.1,.1))
            error=float(np.sum((rv-.5*bias)**2+(rp-.3*bias)**2))
            return (error,bias,memory) if details else error
        lo,hi=bounds[car,4:6]
        if lo==hi:rho=float(lo)
        else:
            grid=np.linspace(lo,hi,65);values=np.array([evaluate(r) for r in grid]);i=int(values.argmin())
            a,b=grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]
            result=minimize_scalar(evaluate,bounds=(a,b),method='bounded',options={'xatol':1e-13})
            rho=min((float(grid[i]),float(result.x)),key=evaluate)
        error,bias,memory=evaluate(rho,True)
        rhos.append(rho);biases.append(bias);memories.append(memory);errors.append(error);usable.append(int(keep.sum()))
    return dict(retention=np.array(rhos),bias=np.array(biases),memory=np.array(memories),
        fit_squared_error=np.array(errors),usable_windows=usable,observed_windows=len(u),
        fitted_future_bias_is_hypothesis=True,parameter_posterior_probability=False,
        uses_true_parameters=False,uses_future_innovations=False)
