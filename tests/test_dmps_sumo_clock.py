import pytest

from coopgate.dmps_sumo_clock import AbsoluteStepConnection


class Fake:
    def __init__(self,broken=False):
        self.simulation=self;self.time=.6;self.calls=[];self.broken=broken
    def getTime(self):return self.time
    def simulationStep(self,target):
        self.calls.append(target)
        if not self.broken:self.time=target
        return 'result'


def test_default_steps_use_current_native_clock_after_backward_reload():
    raw=Fake();conn=AbsoluteStepConnection(raw,.1)
    assert conn.simulationStep()=='result'
    assert raw.time==pytest.approx(.7)
    raw.time=.1
    conn.simulationStep();assert raw.time==pytest.approx(.2)
    conn.simulationStep();assert raw.time==pytest.approx(.3)


def test_native_failure_to_advance_is_not_silently_counted():
    with pytest.raises(RuntimeError):AbsoluteStepConnection(Fake(True),.1).simulationStep()


def test_explicit_targets_are_forwarded_unchanged():
    raw=Fake();AbsoluteStepConnection(raw,.1).simulationStep(3.5)
    assert raw.calls==[3.5]
