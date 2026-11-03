# Three learned roles in training and deployment

Read only the complete independently replayed320-case development pilot after
actual owner exit0. No live formal effects,new SUMO,policy update or risk test.
All three deployed checkpoints match their original full2097152-step joint
MAPPO audits,including on-policy proposal likelihood and independently checked
GAE. Every role has nonzero parameter change. The central critic is training
only;three independently parameterized actors make local proposals.

| Model | Front changed control steps | Merge changed control steps | Rear changed control steps | Steps changing at least two roles |
|---|---:|---:|---:|---:|
|4703|22|73|25|14|
|4704|27|71|20|15|
|4705|25|73|25|16|

Counts are actual in-route command differences from the identical baseline,
including accepted candidate compositions. They are not merely actor branch
entries. Per-model episodes with any role change are83/86/83 of320;per-control
step adoption is a different,smaller quantity and must not be mislabeled as
episode adoption. All roles also produce multiple distinct active proposals.

The declared PPO configurations are identical across the three model protocols:
actor_lr1e-4,ego_actor_lr1e-5,critic_lr3e-4. The1e-4/1e-5/1e-4 rates apply to
the three roles,not different seed runs. Initialization ancestry and role
priors remain disclosed. Different recorded training commits are retained in
their source snapshots;configuration equality alone is not a complete claim
of identical implementation histories.

These findings establish genuine all-role joint training and measured role
participation in executed commands. They do not establish a causal advantage
from cooperation,universal raw MAPPO superiority,or central-critic superiority
over IPPO. Those remain separate experimental questions. Raw report and
actual native receipt are under deployed_role_participation_20261103;source
is tools/audit_deployed_mappo_roles.py.
