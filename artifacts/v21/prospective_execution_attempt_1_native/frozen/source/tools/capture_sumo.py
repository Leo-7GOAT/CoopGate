"""Capture a real SUMO-GUI frame using SUMO's own screenshot API."""
import argparse
from pathlib import Path
from coopgate.sim import SumoEnv

p=argparse.ArgumentParser()
p.add_argument('--out',default='artifacts/sumo_gui.png')
a=p.parse_args()
out=Path(a.out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
with SumoEnv(method='traditional',gui=True) as env:
    env.reset(20000)
    for _ in range(6):
        env.step([4,4,4])
    env.connection.gui.screenshot('View #0',str(out),width=1400,height=700)
    env.step([4,4,4])
if not out.exists():
    raise RuntimeError('SUMO did not create the requested screenshot.')
print(out)
