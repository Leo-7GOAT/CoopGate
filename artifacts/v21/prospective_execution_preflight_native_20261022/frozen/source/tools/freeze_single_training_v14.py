"""Preregister fresh single-ego training, test initials and acceptance criteria."""
import hashlib,json,platform,subprocess
from pathlib import Path
import numpy as np
import torch
import numba

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/v14/single'


def freeze():
    if D.exists():raise RuntimeError('V14 single preregistration already exists')
    sources=[str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'src/coopgate').glob('*.py')]
    sources+=['tools/train_policy_iteration_v14.py','tools/average_policy_v14.py',
              'tools/train_single_pipeline_v14.py','tools/freeze_single_training_v14.py']
    teacher='artifacts/v13/ego/seed_701/policy.pt'
    rng=np.random.default_rng(202610164501)
    seeds=rng.choice(np.arange(45000000,46000000),512,replace=False)
    protocol=dict(version='v14-single-preregistered',training_seeds=[901,911,921],
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        objective='Establish standalone ego learning superiority before any MARL expansion',
        policy='Local neural action selection plus unchanged shared safety filter; no online performance rollout',
        algorithm='Training-only rollout approximate policy iteration; existing method family, not a new RL algorithm',
        width=256,teacher=teacher,teacher_sha256=hashlib.sha256((ROOT/teacher).read_bytes()).hexdigest(),
        phase_a=dict(iterations=6,states=32768,epochs=60,latest_only=False),
        phase_b=dict(iterations=6,states=32768,epochs=80,latest_only=True,seed_offset=100),
        final_policy=dict(method='uniform_parameter_average',trailing_iterations=3),
        model_selection='Same fixed two-stage budget and final three-iteration average for every seed; no test-based checkpoint selection',
        test_initials=[dict(seed=int(s),regime='nominal' if i%2==0 else 'dense') for i,s in enumerate(seeds)],
        disturbances=['clean','uniform','bias','pulse'],
        primary_methods=[f'learned_{s}' for s in (901,911,921)],
        secondary_methods=[f'last_{s}' for s in (901,911,921)],
        baseline='Unmodified ego RolloutGate from V13',cost='own arrival + 0.05 own command variation until own arrival',
        criteria=dict(ego_cost_gain_percent_min=1.,ego_time_gain_percent_min=.5,ego_variation_gain_percent_min=10.,
                      every_seed_cost_improves=True,every_seed_time_improves=True,
                      cost_cluster_ci95_lower_gt_zero=True,time_cluster_ci95_lower_gt_zero=True,
                      all_complete_no_safety_events=True,standalone_inference=True),
        bootstrap=dict(seed=202610164502,draws=10000,cluster='initial, averaging four modes and three fixed trained seeds',interval=.95),
        interpretation=['Training model transitions and actual SUMO executions are counted separately',
                        'Averaging parameters produces one single-ego network; this is not MARL',
                        'No test-set expansion or criterion revision after unblinding',
                        'MARL remains blocked until the full frozen single-ego acceptance and physical/replay audit pass'],
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,numba=numba.__version__),
        training_source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    D.mkdir(parents=True)
    (D/'training_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(training_seeds=protocol['training_seeds'],test_initials=len(seeds),
                         planned_test_runs=len(seeds)*4*7,criteria=protocol['criteria'])),flush=True)


if __name__=='__main__':freeze()
