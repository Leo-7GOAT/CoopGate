import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import plan_history_formal_power as power


def test_stress_projection_distinguishes_unambiguous_safe_and_harmful_fixtures():
    plants=np.array(power.e.design.PLANTS*20);a=np.ones((80,3),bool)
    g=np.tile(np.linspace(1.,3.,80)[:,None],(1,3));h=np.zeros_like(a)
    good=power.simulate(a,h,g,plants,3072,1024,replicates=16)
    assert good['synthetic_joint_passes']==16
    bad=power.simulate(a,~h,g,plants,3072,1024,replicates=16)
    assert bad['synthetic_joint_passes']==0 and bad['synthetic_risk_pass_fraction']==0
    constant=power.simulate(a,h,np.ones_like(g)*2,plants,3072,1024,replicates=16)
    assert constant['synthetic_gain_pass_fraction']==0


def test_power_analysis_cannot_read_live_pilot_effects(monkeypatch):
    reads=[]
    def read(path):
        reads.append(Path(path).name)
        assert Path(path).name=='owner_native_exit.json'
        return dict(actual_native_exit_code=None)
    monkeypatch.setattr(power.e,'read',read)
    with pytest.raises(ValueError,match='owner'):power.run()
    assert reads==['owner_native_exit.json']
