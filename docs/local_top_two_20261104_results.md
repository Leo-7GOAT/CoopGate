# Complete bounded local top two candidate development

All100original cases and three fixed MAPPO actor seeds remain. The600candidate
observations comprise594new actual SUMO runs plus6completed preflights.
The1000completed reference observations are reused, giving1600effective paired
observations and ZERO new independent initials. The previous fresh80are now
reused development data. No statistical error or formal6is allocated.

| Source cohort | Variant | Model | Adoption | Harmful adopted | Mean gain | P95 ms | Max ms | >500ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Prior80 | history | 4703 | 18/80 | 0 | 0.684554% | 181.543 | 1523.141 | 33 |
| Prior80 | history | 4704 | 18/80 | 0 | 0.710413% | 225.126 | 1409.569 | 39 |
| Prior80 | history | 4705 | 19/80 | 0 | 0.729020% | 191.106 | 1484.025 | 30 |
| Prior80 | joint | 4703 | 18/80 | 0 | 0.684554% | 161.811 | 1482.110 | 27 |
| Prior80 | joint | 4704 | 20/80 | 0 | 0.719911% | 166.698 | 1666.171 | 33 |
| Prior80 | joint | 4705 | 19/80 | 0 | 0.729020% | 150.440 | 1386.983 | 27 |
| Prior80 | reserve | 4703 | 16/80 | 0 | 0.613295% | 121.243 | 1015.556 | 14 |
| Prior80 | reserve | 4704 | 17/80 | 0 | 0.668700% | 126.714 | 1108.326 | 16 |
| Prior80 | reserve | 4705 | 17/80 | 0 | 0.636484% | 113.290 | 1058.366 | 12 |
| Prior80 | top_history | 4703 | 19/80 | 0 | 0.725962% | 123.199 | 916.339 | 6 |
| Prior80 | top_history | 4704 | 25/80 | 0 | 0.777588% | 123.587 | 1210.616 | 10 |
| Prior80 | top_history | 4705 | 21/80 | 0 | 0.772953% | 108.112 | 891.663 | 9 |
| Prior80 | top_reserve | 4703 | 16/80 | 0 | 0.613295% | 114.801 | 903.945 | 9 |
| Prior80 | top_reserve | 4704 | 20/80 | 0 | 0.676613% | 125.144 | 1129.356 | 9 |
| Prior80 | top_reserve | 4705 | 18/80 | 0 | 0.639010% | 111.010 | 869.075 | 6 |
| Known20 | history | 4703 | 14/20 | 6 | 1.714526% | 448.550 | 1817.360 | 22 |
| Known20 | history | 4704 | 9/20 | 6 | 1.394002% | 540.615 | 2378.792 | 26 |
| Known20 | history | 4705 | 12/20 | 6 | 1.217007% | 442.789 | 1620.434 | 21 |
| Known20 | joint | 4703 | 15/20 | 6 | 1.754130% | 360.988 | 1510.829 | 19 |
| Known20 | joint | 4704 | 13/20 | 6 | 1.528683% | 483.823 | 1819.633 | 20 |
| Known20 | joint | 4705 | 14/20 | 6 | 1.290000% | 394.142 | 1543.382 | 20 |
| Known20 | reserve | 4703 | 13/20 | 5 | 0.960311% | 326.102 | 1243.996 | 12 |
| Known20 | reserve | 4704 | 12/20 | 4 | 1.100524% | 359.724 | 1427.233 | 13 |
| Known20 | reserve | 4705 | 11/20 | 5 | 0.746189% | 299.197 | 1275.999 | 12 |
| Known20 | top_history | 4703 | 15/20 | 5 | 2.450055% | 276.110 | 1046.415 | 6 |
| Known20 | top_history | 4704 | 14/20 | 4 | 2.087305% | 317.791 | 1324.912 | 9 |
| Known20 | top_history | 4705 | 14/20 | 4 | 2.223520% | 281.977 | 1054.054 | 5 |
| Known20 | top_reserve | 4703 | 13/20 | 4 | 1.881612% | 294.031 | 1059.295 | 8 |
| Known20 | top_reserve | 4704 | 12/20 | 3 | 1.563504% | 323.511 | 1398.943 | 12 |
| Known20 | top_reserve | 4705 | 11/20 | 4 | 1.594463% | 288.914 | 1048.335 | 7 |

The complete development promotion screen was fixed before collection:
all three prior80mean gains >=history+0.05percentage points, no adoption
loss, no failures, no harm on prior80, <=14known model-initial harms and
no newly introduced known harm versus history. It is not a risk certificate.

top_history: FAILS development screen.
```json
{
  "passed": false,
  "prior80_gain_change_pp": {
    "4703": 0.04140786749482417,
    "4704": 0.06717458526692899,
    "4705": 0.043933120020076766
  },
  "prior80_adoption_change": {
    "4703": 1,
    "4704": 7,
    "4705": 2
  },
  "execution_failures": 0,
  "prior80_harm": 0,
  "known20_harm": 13,
  "known20_harm_introduced_vs_history": 0,
  "statistical_certificate": false
}
```
top_reserve: FAILS development screen.
```json
{
  "passed": false,
  "prior80_gain_change_pp": {
    "4703": -0.07125949408683563,
    "4704": -0.03379985245001205,
    "4705": -0.09001083730626402
  },
  "prior80_adoption_change": {
    "4703": -2,
    "4704": 2,
    "4705": -1
  },
  "execution_failures": 0,
  "prior80_harm": 0,
  "known20_harm": 11,
  "known20_harm_introduced_vs_history": 0,
  "statistical_certificate": false
}
```

Actual selected rank-two commands (not merely generated candidates):
```json
{
  "top_history:4703": {
    "episodes": 100,
    "rank_two_adopted_episodes": 3,
    "rank_two_control_steps": 4,
    "total_learning_steps": 42,
    "total_steps": 2337
  },
  "top_history:4704": {
    "episodes": 100,
    "rank_two_adopted_episodes": 10,
    "rank_two_control_steps": 10,
    "total_learning_steps": 47,
    "total_steps": 2338
  },
  "top_history:4705": {
    "episodes": 100,
    "rank_two_adopted_episodes": 7,
    "rank_two_control_steps": 7,
    "total_learning_steps": 46,
    "total_steps": 2339
  },
  "top_reserve:4703": {
    "episodes": 100,
    "rank_two_adopted_episodes": 3,
    "rank_two_control_steps": 4,
    "total_learning_steps": 36,
    "total_steps": 2345
  },
  "top_reserve:4704": {
    "episodes": 100,
    "rank_two_adopted_episodes": 6,
    "rank_two_control_steps": 6,
    "total_learning_steps": 37,
    "total_steps": 2347
  },
  "top_reserve:4705": {
    "episodes": 100,
    "rank_two_adopted_episodes": 5,
    "rank_two_control_steps": 5,
    "total_learning_steps": 36,
    "total_steps": 2348
  }
}
```

Every new episode has complete controller and physics replay. Successful
zero-adoption fallback trajectories exactly equal baseline in402cases.
All earlier failed owners and unfavorable methods remain. Different batches
do not establish causal timing speedups. Scores are finite forecast margins;
actor logits are preferences, not safety probabilities. No communication
latency metric, general raw-MAPPO superiority or hard-real-time claim follows.
