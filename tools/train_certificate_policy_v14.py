"""Offline approximate policy iteration with physics-conditioned action values."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from coopgate.ego_v13 import EgoVector
from coopgate.certificate_policy_v14 import CertificatePolicy,batch_returns

ROOT=Path(__file__).resolve().parents[1]


def fit(model,features,costs,rng,epochs):
    mask=np.isfinite(costs)
    regret=np.where(mask,costs-costs.min(-1,keepdims=True),0.)
    # Cost-sensitive classification preserves equivalence of projected actions.
    target=np.exp(np.clip(-regret/.025,-50,0))*mask
    target/=target.sum(-1,keepdims=True)
    fe=torch.tensor(features);ma=torch.tensor(mask);ta=torch.tensor(target,dtype=torch.float32);re=torch.tensor(regret,dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    for epoch in range(epochs):
        for ids in np.array_split(rng.permutation(len(fe)),max(1,(len(fe)+255)//256)):
            logits=model(fe[ids]).masked_fill(~ma[ids],-1e9)
            loss=(-(ta[ids]*logits.log_softmax(-1)).sum(-1)+3.*(logits.softmax(-1)*re[ids]).sum(-1)).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
    selected=model.choose(features,mask)
    return float(loss.detach()),float(regret[np.arange(len(fe)),selected].mean())


def train(out,seed=805,iterations=5,states=8192,epochs=50,width=64,rule_only=False,variant='certificate'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    if variant=='kinematic':
        from coopgate.kinematic_policy_v14 import KinematicPolicy as policy_class,batch_returns as return_function
    elif variant=='certificate':policy_class,return_function=CertificatePolicy,batch_returns
    else:raise ValueError('Unknown actor feature variant')
    model=policy_class(width)
    sources=['src/coopgate/certificate_policy_v14.py','tools/train_certificate_policy_v14.py']
    if variant=='kinematic':sources+=['src/coopgate/kinematic_policy_v14.py','tools/train_kinematic_policy_v14.py']
    meta=dict(algorithm=variant+'_feature_policy_iteration_v14',seed=seed,width=width,iterations=iterations,
              states_per_iteration=states,epochs=epochs,rule_only=rule_only,learned_vehicles=[1],
              observation='full current state and command memory of three cooperative vehicles; no future disturbances',
              objective='ego arrival + 0.05 ego command variation until own arrival',
              training='rollout approximate policy iteration; rule initialization and learned feedback tail improvement',
              inference='neural scoring of current safety-projected actions; zero performance rollouts',
              selection='fixed final iteration, not best validation checkpoint',
              feature_variant=variant,
              source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    for name in sources:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    (out/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
    torch.save({**meta,'iteration':0,'model':model.state_dict()},out/'initial.pt')
    history=[];started=time.perf_counter();total=visited=0
    for iteration in range(iterations):
        env=EgoVector(32,35000000+seed*10+iteration,.05)
        weights=model.compiled_weights();features=[];costs=[];xs=[];memories=[];clocks=[]
        for j in range(0,states,env.n):
            f,q,count=return_function(env.x,env.previous,env.steps,env.packed,weights,iteration>0 and not rule_only)
            features.append(f);costs.append(q);xs.append(env.x.copy());memories.append(env.previous.copy());clocks.append(env.steps.copy())
            total+=int(count)
            mask=np.isfinite(q);actor=model.choose(f,mask)
            expert=12-np.argmin(q[:,::-1],axis=-1)
            # Off-policy state coverage: expert and apprentice trajectories,
            # with admissible exploratory actions. No held-out observations.
            choose=np.where(rng.random(env.n)<.5,expert,actor)
            random=(rng.random(mask.shape)-np.where(mask,0.,2.)).argmax(-1)
            choose=np.where(rng.random(env.n)<.15,random,choose)
            env.step(choose[:,None])
        fe,co=np.concatenate(features),np.concatenate(costs);visited+=len(fe)
        np.savez_compressed(out/f'data_{iteration+1}.npz',features=fe,costs=co,
                            states=np.concatenate(xs),previous=np.concatenate(memories),steps=np.concatenate(clocks))
        # Every iteration fits freshly evaluated costs of the current policy;
        # mixing stale return targets from old policies would not be API.
        loss,regret=fit(model,fe,co,rng,epochs)
        row=dict(iteration=iteration+1,visited_states=visited,evaluated_model_transitions=total,
                 imitation_loss=loss,training_action_regret_s=regret,elapsed_s=time.perf_counter()-started)
        history.append(row);print(json.dumps(row),flush=True)
        torch.save({**meta,**row,'model':model.state_dict()},out/f'iteration_{iteration+1}.pt')
        torch.save({**meta,**row,'model':model.state_dict()},out/'policy.pt')
        with (out/'training.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=row,lineterminator='\n');writer.writeheader();writer.writerows(history)
    (out/'metadata.json').write_text(json.dumps({**meta,**history[-1]},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=805)
    p.add_argument('--iterations',type=int,default=5);p.add_argument('--states',type=int,default=8192)
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--width',type=int,default=64);p.add_argument('--rule-only',action='store_true')
    a=p.parse_args();train(a.out,a.seed,a.iterations,a.states,a.epochs,a.width,a.rule_only)
