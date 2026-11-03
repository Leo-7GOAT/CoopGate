"""Exact-rational actuator ambiguity example;no SUMO or statistical inference.

Two distinct admissible strong-lag plants generate identical observations when
bounded innovations vary within a control window. This does not contradict
constant-bias conditional identification or change the frozen deployment gate.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def prove():
    dt=F(1,10);rho_a=F(3,4);rho_b=F(19,25);command=F(1)
    memory_a=memory_b=F(0);p_a=p_b=F(-100);v_a=v_b=F(10)
    records=[]
    for step in range(1,6):
        memory_a=rho_a*memory_a+(1-rho_a)*command
        memory_b=rho_b*memory_b+(1-rho_b)*command
        innovation_a=F(0);innovation_b=memory_a-memory_b
        assert abs(innovation_b)<=F(1,10)
        acceleration_a=memory_a+innovation_a
        acceleration_b=memory_b+innovation_b
        assert acceleration_a==acceleration_b
        v_a+=dt*acceleration_a;v_b+=dt*acceleration_b
        p_a+=dt*v_a;p_b+=dt*v_b
        assert v_a==v_b and p_a==p_b and F(0)<v_a<F(20)
        records.append(dict(step=step,innovation_b_exact=str(innovation_b),
            innovation_b_float=float(innovation_b),position_exact=str(p_a),
            speed_exact=str(v_a),identical_at_every_substep=True))
    # Both retentions belong to the declared strong-lag interval.
    import math
    assert math.exp(-.1/.3)<float(rho_a)<float(rho_b)<math.exp(-.1/.5)
    return dict(complete=True,calculation='Exact rational arithmetic, no sampled statistics',
        rho_a_exact=str(rho_a),rho_b_exact=str(rho_b),initial_memory_exact='0',
        command_exact='1',dt_exact='1/10',innovation_bound_exact='1/10',records=records,
        identical_all_observed_positions_and_speeds=True,
        distinct_hidden_actuators=True,nonconstant_innovation_b=True,
        contradicts_constant_bias_identification=False,
        universal_identification_under_arbitrary_bounded_innovations=False,
        controller_or_frozen_source_changed=False,new_sumo_runs=0,
        new_independent_initials=0,new_error_allocated=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=prove();args.out.mkdir(exist_ok=False)
    with (args.out/'report.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)
