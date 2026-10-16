"""Model-based multi-agent policy iteration with sequential actor updates.

Branch returns are generated from the current joint policy and dynamics, not
external planning labels. Counts distinguish visited states from model branches.
"""
import argparse,csv,hashlib,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from coopgate.cooperative_v15 import JointLearner,CooperativeVector
from coopgate.cooperative_iteration_v16 import unilateral_returns
from coopgate.policy_iteration_v14 import load_policy
from prerequisite_v15 import require_single

ROOT=Path(__file__).resolve().parents[1]


def fit_actor(actor,obs,costs,incumbent,rng,epochs):
    valid=np.isfinite(costs);best=np.min(costs,axis=1)
    regret=np.where(valid,costs-best[:,None],0.)
    incumbent_regret=costs[np.arange(len(costs)),incumbent]-best
    targets=np.exp(np.clip(-regret/.04,-60,0))*valid
    targets/=targets.sum(-1,keepdims=True)
    unchanged=incumbent_regret<=1e-9
    targets[unchanged]=0.;targets[np.flatnonzero(unchanged),incumbent[unchanged]]=1.
    weight=1.+np.minimum(5.,5.*incumbent_regret)
    ob=torch.tensor(obs);mask=torch.tensor(valid);target=torch.tensor(targets,dtype=torch.float32)
    re=torch.tensor(regret,dtype=torch.float32);weights=torch.tensor(weight,dtype=torch.float32)
    optimizer=torch.optim.Adam(actor.parameters(),lr=5e-4)
    for _ in range(epochs):
        for ids in np.array_split(rng.permutation(len(obs)),max(1,(len(obs)+255)//256)):
            logits=actor(ob[ids]).masked_fill(~mask[ids],-1e9);lp=logits.log_softmax(-1)
            objective=-(target[ids]*lp).sum(-1)+2.*(logits.softmax(-1)*re[ids]).sum(-1)
            loss=(objective*weights[ids]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite coupled policy update')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),5.);optimizer.step()
    with torch.no_grad():selected=actor(ob).masked_fill(~mask,-1e9).argmax(-1).numpy()
    return dict(loss=float(loss.detach()),training_action_regret=float(regret[np.arange(len(obs)),selected].mean()),
        improvable_state_fraction=float(np.mean(incumbent_regret>1e-9)),
        available_mean_counterfactual_gain=float(incumbent_regret.mean()),
        changed_training_action_fraction=float(np.mean(selected!=incumbent)))


def train(out,seed=1601,ego_seed=901,cycles=2,states=2048,epochs=60,exploration=.15):
    prerequisite=require_single(ROOT)
    if cycles<1 or states<32 or epochs<1:raise ValueError('Positive complete training budget required')
    if not np.isfinite(exploration) or not 0<=exploration<=1:raise ValueError('Invalid exploration probability')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    teacher=ROOT/f'artifacts/v14/single/seed_{ego_seed}/policy.pt'
    ego,_=load_policy(teacher);model=JointLearner(ego)
    for p in model.critics.parameters():p.requires_grad_(False)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    order=[0,2,1];meta=dict(algorithm='coupled_policy_iteration_v16',seed=seed,ego_seed=ego_seed,
        actor_width=model.width,trainable_vehicles=[0,1,2],actor_update_order=order,
        cycles=cycles,target_updates=3*cycles,states_per_update=states,epochs=epochs,
        training_backend='Known model with hidden disturbances in visitation; nominal counterfactual joint returns',
        objective='Mean individual arrival plus 0.05 individual variation, each stopped at own arrival',
        data='Current joint-policy trajectories with declared admissible exploration of the currently updated role',
        exploration_probability=exploration,
        policy_improvement='One role at a time; all other actors frozen, full future joint policy frozen during each data batch',
        critic='Unused, frozen architecture fields only; no trained value critic in this algorithm',
        labels='Self-generated complete joint-policy returns, no external planning action labels',
        inference='Three local neural proposals and shared physical shield; no performance rollout at execution',
        initial_ego_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),prerequisite=prerequisite,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        checkpoint_selection='Fixed final update, no validation-based checkpoint choice')
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/prerequisite_v15.py']
    meta['source_sha256']={str(p.resolve().relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in meta['source_sha256']:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    (out/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
    torch.save({**meta,'updates':0,'model':model.state_dict()},out/'initial.pt')
    history=[];visited=branches=environment_steps=0;started=time.perf_counter()
    for update in range(3*cycles):
        role=order[update%3];env=CooperativeVector(32,81000000+seed*20+update)
        stored=[];count=0;round_branches=0
        frozen_other={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith(f'actors.{role}.')}
        while count<states:
            obs,_,mask=env.inputs();ids=np.flatnonzero(env.x[:,role,0]<60)[:states-count]
            if len(ids):
                costs,incumbent,evaluated=unilateral_returns(env.x[ids],env.previous[ids],env.steps[ids],model,role)
                stored.append((obs[ids,role].copy(),costs,incumbent,env.x[ids].copy(),env.previous[ids].copy(),env.steps[ids].copy()))
                count+=len(ids);round_branches+=evaluated
            action=model.proposal(obs,mask)
            random=(rng.random(mask[:,role].shape)-np.where(mask[:,role],0.,2.)).argmax(-1)
            action[:,role]=np.where(rng.random(env.n)<exploration,random,action[:,role])
            env.step(action);environment_steps+=env.n
        ob,cost,incumbent,x,prev,clock=[np.concatenate([r[i] for r in stored]) for i in range(6)]
        np.savez_compressed(out/f'data_{update+1}.npz',obs=ob,costs=cost,incumbent=incumbent,states=x,previous=prev,steps=clock)
        stats=fit_actor(model.actors[role],ob,cost,incumbent,rng,epochs)
        assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen_other.items())
        visited+=len(ob);branches+=round_branches
        changes=[sum(float((value-initial[key]).square().sum()) for key,value in model.state_dict().items()
            if key.startswith(f'actors.{i}.')) for i in range(3)]
        row=dict(updates=update+1,role=role,visited_states=visited,joint_environment_steps=environment_steps,
            evaluated_model_transitions=branches,**stats,elapsed_s=time.perf_counter()-started,
            actor0_squared_change=changes[0],actor1_squared_change=changes[1],actor2_squared_change=changes[2])
        history.append(row);print(json.dumps(row),flush=True)
        metadata={**meta,**row,'actor_squared_parameter_change':changes}
        torch.save({**metadata,'model':model.state_dict()},out/f'update_{update+1}.pt')
        torch.save({**metadata,'model':model.state_dict()},out/'policy.pt')
        with (out/'training.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=row,lineterminator='\n');writer.writeheader();writer.writerows(history)
    if not all(v>1e-8 for v in changes):raise RuntimeError('An actor did not update')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=1601)
    p.add_argument('--ego-seed',type=int,default=901);p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--states',type=int,default=2048);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--exploration',type=float,default=.15)
    a=p.parse_args();train(a.out,a.seed,a.ego_seed,a.cycles,a.states,a.epochs,a.exploration)
