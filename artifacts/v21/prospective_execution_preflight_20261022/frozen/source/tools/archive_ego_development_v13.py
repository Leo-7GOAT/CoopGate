"""Keep unsuccessful pilots and reports visible, separate from frozen tests."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v13/development'


if __name__=='__main__':
    D.mkdir(parents=True,exist_ok=False)
    names=['v13_ego_scratch_601','v13_ego_warm_601','v13_ego_memory_ppo_601',
        'v13_ego_dqn_601','v13_ego_memory_dqn_601','v13_ego_dyna_601']
    for name in names:
        source=ROOT/'runs'/name;target=D/name;target.mkdir()
        for filename in ('initial.pt','policy.pt','metadata.json','training.csv'):
            path=source/filename
            if path.exists():shutil.copy2(path,target/filename)
    reports=['v13_ego_development_current.json','v13_ego_more_development.json','v13_ego_dyna_final.json',
        'v13_augmented_development.json','v13_augmented_imitation.json','v13_augmented_q_development.json',
        'v13_ego_baseline_detailed.json','v13_compiled_float32_development.json']
    for name in reports:
        if (ROOT/'runs'/name).exists():shutil.copy2(ROOT/'runs'/name,D/name)
    manifest={'purpose':'development only; none of these seeds or results are counted as frozen independent V13 tests',
        'known_limit':'pilot source evolved between runs; preserve metadata and results without claiming identical-source replay',
        'fresh_release_training_seeds':[701,711,721],
        'negative_findings':['raw PPO and DQN did not beat the strong ego rollout baseline',
            'adding neighbor command memory alone did not establish superiority',
            'complete-action model-based Q training did not establish superiority',
            'two-stage planner diagnostic did not reliably improve the strong one-stage baseline'],
        'sha256':{str(p.relative_to(D)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*') if p.is_file()}}
    (D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'archived_pilots':len(names),'archived_reports':len(reports)}))
