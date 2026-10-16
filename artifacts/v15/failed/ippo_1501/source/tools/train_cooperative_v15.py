"""Development-only MAPPO/IPPO with a mandatory passed single-ego prerequisite.

Three actors update from the same sequential interaction and shared team return.
Actor likelihoods always refer to sampled pre-shield proposals. No teacher action
labels or online performance optimizer are used in these PPO updates.
"""
import argparse,copy,csv,hashlib,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from coopgate.cooperative_v15 import JointLearner,CooperativeVector
from coopgate.policy_iteration_v14 import load_policy
from prerequisite_v15 import require_single

ROOT=Path(__file__).resolve().parents[1]


def train(out,seed=1501,ego_seed=901,steps=262144,critic_kind='central',anchor=.01):
    prerequisite=require_single(ROOT)
    frozen=json.loads((ROOT/'artifacts/v14/single/training_protocol.json').read_text())
    if ego_seed not in frozen['training_seeds']:raise ValueError('Use a preregistered single-ego seed')
    if steps<=0 or steps%32:raise ValueError('Training budget must be a positive multiple of 32 joint steps')
    if not np.isfinite(anchor) or anchor<0:raise ValueError('Invalid anchoring coefficient')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    teacher=ROOT/f'artifacts/v14/single/seed_{ego_seed}/policy.pt'
    ego,_=load_policy(teacher);model=JointLearner(ego,critic_kind)
    reference=copy.deepcopy(model).eval()
    for p in reference.parameters():p.requires_grad_(False)
    # Match proposal/minibatch RNG starts across centralized/local critic runs.
    torch.manual_seed(seed+151500000)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam([dict(params=model.actors.parameters(),lr=1e-4),
                                dict(params=model.critics.parameters(),lr=3e-4)],eps=1e-5)
    n=32;env=CooperativeVector(n,seed+80000000)
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/prerequisite_v15.py']
    sources={str(p.resolve().relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in sources:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    meta=dict(algorithm='mappo_three_actor_v15' if critic_kind=='central' else 'ippo_three_actor_v15',
        purpose='Development experiment; no formal multi-agent acceptance',seed=seed,ego_seed=ego_seed,
        proposal_rng_seed=seed+151500000,
        actor_width=model.width,critic_kind=critic_kind,trainable_vehicles=[0,1,2],steps=0,target_steps=steps,
        n_envs=n,rollout_steps=64,epochs=4,batch_size=256,gamma=1.,gae_lambda=.95,ppo_clip=.2,
        actor_lr=1e-4,critic_lr=3e-4,entropy_weight=.002,anchor_kl_weight=anchor,
        objective='mean individual arrival + 0.05 individual command variation until individual arrival',
        game='partially observed common-payoff cooperative finite-horizon stochastic game',
        training_backend='exact longitudinal model; not SUMO invocations',
        initial_ego_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
        initial_mainroad_actors='highest-admissible proposal preference; all parameters train',
        actor_observation='18-dimensional local kinematics and neighbor command memory + 8 fixed kinematic features',
        disturbance_information='correlated disturbance mode and future error tape are hidden from actors and critics',
        physical_filter='same centralized certified 27-vertex projection; bounded error 0.5',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=sources,prerequisite=prerequisite,checkpoint_selection='fixed final training budget')
    meta['actor_parameter_delta_l2_definition']='sum of squared parameter changes (squared L2), not its square root'
    (out/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
    torch.save({**meta,'model':model.state_dict()},out/'initial.pt')
    total=active_samples=0;history=[];episodes=[];started=time.perf_counter()
    while total<steps:
        records=[]
        for _ in range(min(64,(steps-total)//n)):
            obs,state,mask=env.inputs();ob=torch.tensor(obs);st=torch.tensor(state);ma=torch.tensor(mask)
            with torch.no_grad():
                dist=model.distribution(ob,ma);action=dist.sample();lp=dist.log_prob(action);value=model.value(ob,st)
            nxt,reward,boundary,active,finished=env.step(action.numpy());episodes.extend(finished)
            with torch.no_grad():next_value=model.value(torch.tensor(nxt[0]),torch.tensor(nxt[1]))
            records.append((obs,state,mask,action.numpy(),lp.numpy(),value.numpy(),reward,next_value.numpy(),boundary,active))
            total+=n;active_samples+=int(active.sum())
        obs,state,mask,actions,oldlp,values,rewards,nextvalues,boundaries,alive=map(np.asarray,zip(*records))
        advantages=np.zeros_like(values);carry=np.zeros((n,3))
        for t in range(len(records)-1,-1,-1):
            continuation=(~boundaries[t])[:,None]
            delta=rewards[t,:,None]+continuation*nextvalues[t]-values[t]
            carry=delta+.95*continuation*carry;advantages[t]=carry
        returns=advantages+values
        advantages=(advantages-advantages[alive].mean())/(advantages[alive].std()+1e-8)
        b=len(records)*n
        ob,st,ma,ac,ol,ad,re,al=[torch.as_tensor(z.reshape((b,)+z.shape[2:])) for z in
                               (obs,state,mask,actions,oldlp,advantages,returns,alive)]
        for epoch in range(4):
            for ids in torch.randperm(b).split(256):
                dist=model.distribution(ob[ids],ma[ids]);ratio=(dist.log_prob(ac[ids])-ol[ids]).exp()
                objective=torch.minimum(ratio*ad[ids],ratio.clamp(.8,1.2)*ad[ids])
                actor=-objective[al[ids]].mean();entropy=dist.entropy()[al[ids]].mean()
                with torch.no_grad():ref=reference.distribution(ob[ids],ma[ids])
                kl=torch.distributions.kl_divergence(ref,dist)[al[ids]].mean()
                value_loss=(model.value(ob[ids],st[ids])-re[ids]).square().mean()
                loss=actor+.5*value_loss-.002*entropy+anchor*kl
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite joint PPO loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        recent=episodes[-100:]
        changes=[sum(float((value-initial[key]).square().sum()) for key,value in model.state_dict().items()
                     if key.startswith(f'actors.{i}.')) for i in range(3)]
        row=dict(joint_environment_steps=total,active_actor_samples=active_samples,episodes=env.episodes,
            team_cost_last100=float(np.mean([r['cost'] for r in recent])) if recent else 25.,
            actor_loss=float(actor.detach()),critic_loss=float(value_loss.detach()),entropy=float(entropy.detach()),
            reference_kl=float(kl.detach()),elapsed_s=time.perf_counter()-started,
            actor0_delta_l2=changes[0],actor1_delta_l2=changes[1],actor2_delta_l2=changes[2])
        history.append(row)
        metadata={**meta,'steps':total,'episodes':env.episodes,
            'active_actor_samples':active_samples,'actor_parameter_delta_l2':changes}
        torch.save({**metadata,'model':model.state_dict()},out/'policy.pt')
        if total%65536==0 or total==steps:
            print(json.dumps(dict(seed=seed,**row)),flush=True)
            torch.save({**metadata,'model':model.state_dict()},out/f'step_{total}.pt')
        with (out/'training.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=row,lineterminator='\n');writer.writeheader();writer.writerows(history)
    if not all(v>1e-8 for v in changes):raise RuntimeError('An actor did not update')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'episodes.json').write_text(json.dumps(episodes,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=1501)
    p.add_argument('--ego-seed',type=int,default=901);p.add_argument('--steps',type=int,default=262144)
    p.add_argument('--critic',choices=['central','local'],default='central');p.add_argument('--anchor',type=float,default=.01)
    a=p.parse_args();train(a.out,a.seed,a.ego_seed,a.steps,a.critic,a.anchor)
