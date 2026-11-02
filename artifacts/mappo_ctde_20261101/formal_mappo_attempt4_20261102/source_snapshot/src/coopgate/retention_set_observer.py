"""Online set membership for unit-gain,zero-drag actuator lag classes.

Only measured decision-boundary positions/speeds and EXECUTED commands enter.
Cells are OUTER parameter sets,not a posterior or probability. This observer
does not change the physical baseline or certify candidate performance.
"""
import numpy as np
from .actuator_recovery_v34 import class_bounds


def down(x):return np.nextafter(x,-np.inf)
def up(x):return np.nextafter(x,np.inf)


def multiply(alo,ahi,blo,bhi):
    terms=np.stack(np.broadcast_arrays(alo*blo,alo*bhi,ahi*blo,ahi*bhi))
    return down(terms.min(axis=0)),up(terms.max(axis=0))


class RetentionSetObserver:
    def __init__(self,profile,cells=32):
        if profile not in ('lag_mild','lag_strong') or cells!=32:
            raise ValueError('Registered32-cell observer is only for known unit-gain zero-drag lag classes')
        self.profile=profile;bounds=class_bounds(profile)
        edges=np.stack([np.linspace(b[4],b[5],cells+1) for b in bounds])
        self.lower,self.upper=edges[:,:-1].copy(),edges[:,1:].copy()
        self.feasible=np.ones((3,cells),bool)
        self.memory_lower=np.zeros((3,cells));self.memory_upper=np.zeros((3,cells))
        self.radius=bounds[:,6,None];self.last_state=None;self.commands=[]

    def observe(self,state,executed_previous):
        x=np.asarray(state,dtype=float);u=np.asarray(executed_previous,dtype=float)
        if x.shape!=(3,2) or u.shape!=(3,) or not np.isfinite(x).all() or not np.isfinite(u).all():
            raise ValueError('Finite measured3-car state and actual previous command required')
        if np.any(x[:,1]<0) or np.any(x[:,1]>20) or np.any(u<-5) or np.any(u>2.5):
            raise ValueError('Observer input outside registered kinematic/command bounds')
        if self.last_state is None:
            if np.any(u):raise ValueError('Observer reset requires actual zero initial actuator command')
            self.last_state=x.copy();return self.status()
        vlo=self.last_state[:,1,None].copy();vhi=vlo.copy()
        plo=self.last_state[:,0,None].copy();phi=plo.copy();command=u[:,None]
        for _ in range(5):
            a,b=multiply(self.lower,self.upper,self.memory_lower,self.memory_upper)
            c,d=multiply(down(1.-self.upper),up(1.-self.lower),command,command)
            self.memory_lower,self.memory_upper=down(a+c),up(b+d)
            alo,ahi=down(self.memory_lower-self.radius),up(self.memory_upper+self.radius)
            vlo=np.clip(down(vlo+down(.1*alo)),0.,20.)
            vhi=np.clip(up(vhi+up(.1*ahi)),0.,20.)
            plo=down(plo+down(.1*vlo));phi=up(phi+up(.1*vhi))
        # Outward interval arithmetic already covers floating roundoff. This
        # additional fixed tolerance avoids treating tiny sensor serialization
        # differences as evidence,never makes a parameter cell easier to reject.
        tolerance=1e-9
        compatible=(x[:,1,None]>=vlo-tolerance)&(x[:,1,None]<=vhi+tolerance)
        compatible&=(x[:,0,None]>=plo-tolerance)&(x[:,0,None]<=phi+tolerance)
        self.feasible&=compatible;self.last_state=x.copy();self.commands.append(u.copy())
        return self.status()

    def retention_bounds(self):
        if not self.feasible.any(axis=1).all():raise ValueError('No declared lag parameter cell matches the observed history')
        return (np.min(np.where(self.feasible,self.lower,np.inf),axis=1),
                np.max(np.where(self.feasible,self.upper,-np.inf),axis=1))

    def status(self):
        return dict(consistent=bool(self.feasible.any(axis=1).all()),feasible_cells=self.feasible.sum(axis=1).tolist(),
            intervals_observed=len(self.commands),uses_true_parameters=False,uses_future_innovations=False,
            parameter_posterior_probability=False)
