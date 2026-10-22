"""Single-worker frozen SUMO study; retain failures and paired perturbations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import numba
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.experiment_v11 import prepare
from coopgate.learning_v6 import load_models
from coopgate.merge_v4 import MergeConfig

ROOT = Path(__file__).resolve().parents[1]; D = ROOT/'artifacts/v11'


def run(out,protocol_path=D/'protocol.json'):
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    models = load_models(ROOT/'artifacts/v6', protocol['model_seeds'])
    config = MergeConfig(**protocol['config']); radius = protocol['radius']; methods = protocol['methods']
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for method in methods:
        gate, _ = prepare(method, models, warm, np.zeros(3), config, radius, 0)
        gate.select(warm, np.zeros(3), np.full(3, config.accel))
    rows, timings, initials = [], [], []
    with DisturbedMergeEnv(config, 'constant') as env:
        for regime, seeds in protocol['test_seeds'].items():
            for index, seed in enumerate(seeds):
                for mode_index, mode in enumerate(protocol['disturbances']):
                    order = np.random.default_rng(seed+1103+mode_index).permutation(methods).tolist()
                    for order_index, method in enumerate(order):
                        env.reset_disturbed(seed, regime, mode, radius)
                        if mode_index == 0 and order_index == 0:
                            initials.append({'seed': seed, 'scenario': regime, 'state': env.x.tolist()})
                        start = time.perf_counter()
                        info = {'selected': -1, 'proposal': -1, 'initial_v9_base_cost_s': config.horizon}
                        cycle = []
                        try:
                            env.gate, info = prepare(method, models, env.x, env.prev, config, radius, seed)
                        except (ValueError, RuntimeError) as exc:
                            env.abort_reason = 'initialization: '+str(exc)
                        selection_ms = (time.perf_counter()-start)*1000
                        if not env.abort_reason:
                            while True:
                                start = time.perf_counter(); _, _, done, truncated, _ = env.step([4]*3)
                                cycle.append((time.perf_counter()-start)*1000)
                                if done or truncated: break
                        trace = env.trace; metrics = env.metrics()
                        commands=np.array([[0.,0.,0.]]+[t['command'] for t in trace])
                        changes=np.diff(commands,axis=0)
                        total_variation=float(np.abs(changes).sum())
                        rows.append({**metrics, 'method': method, **info, 'execution_order': order_index,
                            'total_command_variation':total_variation, 'trace_steps':len(trace),
                            'peak_command_change_rate':float(np.max(np.abs(changes))/config.decision_dt) if len(changes) else 0.,
                            'selection_ms': selection_ms, 'first_response_ms': selection_ms+(cycle[0] if cycle else 0.),
                            'stopped_vehicle_seconds': config.decision_dt*sum(sum(x[1]<.2 and x[0]<config.goal for x in t['x']) for t in trace),
                            'accepted_updates': sum(t.get('accepted_updates',0) for t in trace),
                            'base_switches': sum(trace[k].get('active_profile',-1)!=trace[k-1].get('active_profile',-1) for k in range(1,len(trace))),
                            'tail_feedback_steps': sum(t.get('tail_feedback_steps',0) for t in trace),
                            'trial_checks': sum(t.get('trial_checks',0) for t in trace),
                            'pruned_trials': sum(t.get('pruned_trials',0) for t in trace),
                            'variation_tie_improvement_steps':sum(t.get('selected_ticks',0)==t.get('base_ticks',0) and t.get('selected_variation_units',0)<t.get('base_variation_units',0) for t in trace),
                            'family_switches':sum(trace[k].get('active_family',-1)!=trace[k-1].get('active_family',-1) for k in range(1,len(trace))),
                            'positive_gain_steps': sum(t.get('selected_ticks',0)<t.get('base_ticks',0) for t in trace)})
                        timings.append({'seed': seed, 'scenario': regime, 'disturbance': mode, 'method': method,
                                        'cycle_ms': cycle, 'controller_ms': [t['gate_ms'] for t in trace]})
                        if index == 0:
                            (out/f'trace_{regime}_{mode}_{method}.json').write_text(json.dumps(trace,indent=2)+'\n',newline='\n')
                if (index+1) % 5 == 0:
                    print(json.dumps({'regime':regime,'scenes':index+1,'executions':len(rows)}),flush=True)
                    with (out/'episodes.csv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    # Always write the final partial batch as well.
    with (out/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for name,value in [('initial_states.json',initials),('timing.json',timings)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n',newline='\n')
    manifest={'episodes':len(rows),'python':platform.python_version(),'platform':platform.platform(),
              'numpy':np.__version__,'numba':numba.__version__,'single_worker':True,
              'source_sha256':{str(protocol_path.resolve().relative_to(ROOT)).replace('\\','/'):hashlib.sha256(protocol_path.read_bytes()).hexdigest()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('runs/v11_reproduction'))
    parser.add_argument('--protocol',type=Path,default=D/'protocol.json')
    args=parser.parse_args();run(args.out,args.protocol)
