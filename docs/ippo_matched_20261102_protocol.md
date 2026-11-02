# Fixed local-critic PPO comparator for the MAPPO family

Register seeds4703/4704/4705,each2,097,152 new joint on-policy steps. Same I
actor parent,three independent ReLU256 local actors,rear-coasting prior,
proposal/environment streams,shared team reward,LR[1e-4,1e-5,1e-4],critic3e-4,
gamma/lambda1,entropy.01,64 environments/rollout64/five epochs/minibatch1024.
Only critic information changes: the centralized37-input scalar team value
is replaced by a shared26-input LOCAL value function evaluated separately
for each role. This is IPPO with shared local value parameters,not three
independently parameterized critics. Actor capacities are identical; critic
parameter counts and compute differ and must be disclosed,not called equal.
Deployment proposals use no critic. No teacher labels or PI updates enter PPO.

Each paired reference is its already completed MAPPO seed. Use the SAME fresh
development sets95700001/96000001/96100001,128 initials and four modes. Include
rule,both strong planners,I parent,MAPPO,IPPO,untouched IPPO joint initializer,
three role reverts,and highest-admissible rear reference:11 methods/5,632
exact executions per seed. Then88 actual SUMO integration/replay runs. If
zero IPPO failure plus positive team gain AND positive one-sided development
t lower versus strong team planning,run the fixed3,584 full SUMO follow-up;
otherwise retain the negative result and mark that stage not enabled.

All three seeds are required; no replacement,best checkpoint or model choice.
These cases were already used by MAPPO and do not add independent initials.
Comparisons describe the value-information ablation and are not an untouched
risk certificate or from-scratch sample-efficiency result. Ancestor PI training
remains disclosed. Every native handle/exit,source hash,rollout likelihood,GAE
and role update audit remains required. No retry into an existing directory.
The first seed may overlap score-grid review; timings are not isolated.
The other seeds remain registered and must subsequently finish their budgets.
