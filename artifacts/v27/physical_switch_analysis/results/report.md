# Common-condition physical and performance gate comparison

All 18,816 historical executions and 6,912 added physical-gate executions are retained. They reuse 128 development and 160 transfer initials. No new independent efficacy initials or risk certificate.

Adoption means at least one active vehicle executed a command different from the same-state common baseline. Harm means any vehicle cost/arrival exceeds the full-baseline outcome, or execution failure. Zero-adoption conditional risk is undefined; original event counts are retained separately.

All intervals below are pointwise, descriptive paired-initial mean-cost bootstrap intervals. They do not provide simultaneous coverage or conditional-risk bounds. MAMPS/MPS are common-oracle/common-baseline adaptations, not original benchmark reproductions.

## Development: one preassigned actual mode per initial

| Method | Gain % | Gain 95% interval (s) | Adopted / N | Harmful / adopted | Original adopted | Failures | Bundle harmful / adopted |
|---|---:|---|---|---|---:|---:|---|
| baseline | 0.0000 | [0.000000, 0.000000] | 0/128 | 0/0 | 0 | 0 | 0/0 |
| repair_1802 | 1.0368 | [0.062660, 0.153607] | 44/128 | 1/44 | 44 | 0 | 2/92 |
| reference_1802 | 1.0368 | [0.062660, 0.153607] | 44/128 | 1/44 | 44 | 0 | 2/92 |
| phase05_1802 | 0.7572 | [0.041380, 0.116416] | 36/128 | 0/36 | 36 | 0 | 1/85 |
| phase10_1802 | 0.8206 | [0.046466, 0.125087] | 39/128 | 1/39 | 39 | 0 | 2/90 |
| repair_1812 | 1.0650 | [0.065307, 0.156577] | 44/128 | 0/44 | 44 | 0 | 1/94 |
| reference_1812 | 1.0650 | [0.065307, 0.156577] | 44/128 | 0/44 | 44 | 0 | 1/94 |
| phase05_1812 | 1.0287 | [0.062213, 0.152061] | 38/128 | 0/38 | 38 | 0 | 1/84 |
| phase10_1812 | 1.0412 | [0.063435, 0.153446] | 42/128 | 0/42 | 42 | 0 | 1/91 |
| repair_1822 | 1.0039 | [0.059749, 0.148667] | 46/128 | 0/46 | 46 | 0 | 1/93 |
| reference_1822 | 1.0039 | [0.059749, 0.148667] | 46/128 | 0/46 | 46 | 0 | 1/93 |
| phase05_1822 | 0.9664 | [0.055905, 0.144777] | 39/128 | 0/39 | 39 | 0 | 1/84 |
| phase10_1822 | 0.9741 | [0.056783, 0.145823] | 41/128 | 0/41 | 41 | 0 | 1/87 |
| mamps_shared_1802 | 2.4552 | [0.158687, 0.338218] | 128/128 | 70/128 | 128 | 0 | 98/128 |
| mps_shared_1802 | 2.4552 | [0.157552, 0.339364] | 128/128 | 70/128 | 128 | 0 | 98/128 |
| mamps_shared_1812 | 3.0168 | [0.225340, 0.389974] | 128/128 | 74/128 | 128 | 0 | 98/128 |
| mps_shared_1812 | 2.9797 | [0.220785, 0.386637] | 128/128 | 74/128 | 128 | 0 | 97/128 |
| mamps_shared_1822 | 2.7352 | [0.199054, 0.356616] | 128/128 | 69/128 | 128 | 0 | 95/128 |
| mps_shared_1822 | 2.7352 | [0.198730, 0.357428] | 128/128 | 67/128 | 128 | 0 | 94/128 |

The JSON also contains every ordinary/dense subgroup, every vehicle outcome, all-mode failure counts and the exact preassigned mode of every initial.

## Primary paired contrasts

Positive cost increment favors phase05. Negative risk difference favors phase05. A lower risk alone is not a dominance claim if adoption or gain decreases.

| Study | Scene | Plant | Model | Comparator | Cost increment (s) | Pointwise 95% interval | Adoption difference | Conditional risk difference |
|---|---|---|---:|---|---:|---|---:|---:|
| phase | mixture | additive | 1802 | mamps_shared_1802 | -0.171265 | [-0.252895, -0.091512] | -0.7188 | -0.5469 |
| phase | mixture | additive | 1802 | mps_shared_1802 | -0.171265 | [-0.252852, -0.091542] | -0.7188 | -0.5469 |
| phase | mixture | additive | 1812 | mamps_shared_1812 | -0.200521 | [-0.275961, -0.129110] | -0.7031 | -0.5781 |
| phase | mixture | additive | 1812 | mps_shared_1812 | -0.196777 | [-0.271894, -0.124132] | -0.7031 | -0.5781 |
| phase | mixture | additive | 1822 | mamps_shared_1822 | -0.178406 | [-0.248171, -0.112567] | -0.6953 | -0.5391 |
| phase | mixture | additive | 1822 | mps_shared_1822 | -0.178406 | [-0.246279, -0.113485] | -0.6953 | -0.5234 |
| transfer | nominal | nominal | 1802 | mamps_shared_1802 | 0.002441 | [-0.074382, 0.105632] | -0.7500 | -0.4062 |
| transfer | nominal | nominal | 1802 | mps_shared_1802 | 0.001790 | [-0.075033, 0.105147] | -0.7500 | -0.4062 |
| transfer | nominal | nominal | 1812 | mamps_shared_1812 | -0.029134 | [-0.084639, 0.022957] | -0.7812 | -0.6250 |
| transfer | nominal | nominal | 1812 | mps_shared_1812 | -0.028483 | [-0.083993, 0.023930] | -0.7812 | -0.6250 |
| transfer | nominal | nominal | 1822 | mamps_shared_1822 | 0.014811 | [-0.054203, 0.097168] | -0.7812 | -0.5000 |
| transfer | nominal | nominal | 1822 | mps_shared_1822 | 0.021322 | [-0.048991, 0.103194] | -0.7812 | -0.5000 |
| transfer | nominal | gain_drag | 1802 | mamps_shared_1802 | -0.029215 | [-0.141852, 0.086764] | -0.8125 | -0.4688 |
| transfer | nominal | gain_drag | 1802 | mps_shared_1802 | -0.029867 | [-0.142259, 0.086593] | -0.8125 | -0.4688 |
| transfer | nominal | gain_drag | 1812 | mamps_shared_1812 | -0.085775 | [-0.190104, -0.009599] | -0.8125 | -0.4688 |
| transfer | nominal | gain_drag | 1812 | mps_shared_1812 | -0.085775 | [-0.189945, -0.009766] | -0.8125 | -0.4688 |
| transfer | nominal | gain_drag | 1822 | mamps_shared_1822 | -0.021810 | [-0.120117, 0.071289] | -0.9062 | -0.5312 |
| transfer | nominal | gain_drag | 1822 | mps_shared_1822 | -0.016927 | [-0.115885, 0.076172] | -0.9062 | -0.5312 |
| transfer | nominal | lag_mild | 1802 | mamps_shared_1802 | -0.003743 | [-0.089355, 0.102702] | -0.8438 | -0.4375 |
| transfer | nominal | lag_mild | 1802 | mps_shared_1802 | -0.004395 | [-0.089844, 0.102051] | -0.8438 | -0.4375 |
| transfer | nominal | lag_mild | 1812 | mamps_shared_1812 | -0.047689 | [-0.126636, 0.011068] | -0.8438 | -0.6250 |
| transfer | nominal | lag_mild | 1812 | mps_shared_1812 | -0.048340 | [-0.126953, 0.009603] | -0.8438 | -0.6250 |
| transfer | nominal | lag_mild | 1822 | mamps_shared_1822 | 0.010417 | [-0.073901, 0.097990] | -0.9062 | -0.5312 |
| transfer | nominal | lag_mild | 1822 | mps_shared_1822 | 0.017578 | [-0.068042, 0.105469] | -0.9062 | -0.5312 |
| transfer | nominal | lag_strong | 1802 | mamps_shared_1802 | -0.144531 | [-0.239099, -0.037598] | -0.7812 | -0.0893 |
| transfer | nominal | lag_strong | 1802 | mps_shared_1802 | -0.142578 | [-0.236820, -0.035966] | -0.7812 | -0.0893 |
| transfer | nominal | lag_strong | 1812 | mamps_shared_1812 | -0.203288 | [-0.311047, -0.113924] | -0.7500 | -0.0625 |
| transfer | nominal | lag_strong | 1812 | mps_shared_1812 | -0.203288 | [-0.311047, -0.113924] | -0.7500 | -0.0625 |
| transfer | nominal | lag_strong | 1822 | mamps_shared_1822 | -0.177734 | [-0.276538, -0.090820] | -0.7188 | -0.3266 |
| transfer | nominal | lag_strong | 1822 | mps_shared_1822 | -0.173828 | [-0.272469, -0.088053] | -0.7188 | -0.3266 |
| transfer | dense | nominal | 1802 | mamps_shared_1802 | -0.043620 | [-0.186202, 0.092778] | -0.8125 | -0.5521 |
| transfer | dense | nominal | 1802 | mps_shared_1802 | -0.041016 | [-0.184412, 0.094727] | -0.8125 | -0.5521 |
| transfer | dense | nominal | 1812 | mamps_shared_1812 | -0.114095 | [-0.261393, 0.019702] | -0.9062 | -0.7812 |
| transfer | dense | nominal | 1812 | mps_shared_1812 | -0.114746 | [-0.261397, 0.018884] | -0.9062 | -0.7812 |
| transfer | dense | nominal | 1822 | mamps_shared_1822 | -0.076823 | [-0.248535, 0.098149] | -0.9062 | -0.6875 |
| transfer | dense | nominal | 1822 | mps_shared_1822 | -0.070964 | [-0.243018, 0.102702] | -0.9062 | -0.6875 |
| transfer | dense | gain_drag | 1802 | mamps_shared_1802 | -0.063151 | [-0.197933, 0.063802] | -0.7500 | -0.6250 |
| transfer | dense | gain_drag | 1802 | mps_shared_1802 | -0.066406 | [-0.201172, 0.059896] | -0.7500 | -0.6250 |
| transfer | dense | gain_drag | 1812 | mamps_shared_1812 | -0.088216 | [-0.230802, 0.052409] | -0.8125 | -0.6562 |
| transfer | dense | gain_drag | 1812 | mps_shared_1812 | -0.088542 | [-0.231136, 0.052083] | -0.8125 | -0.6562 |
| transfer | dense | gain_drag | 1822 | mamps_shared_1822 | -0.075195 | [-0.239583, 0.102222] | -0.7812 | -0.5312 |
| transfer | dense | gain_drag | 1822 | mps_shared_1822 | -0.089518 | [-0.249683, 0.085620] | -0.7812 | -0.5312 |
| transfer | dense | lag_mild | 1802 | mamps_shared_1802 | -0.126628 | [-0.299487, 0.038411] | -0.9062 | -0.6875 |
| transfer | dense | lag_mild | 1802 | mps_shared_1802 | -0.126628 | [-0.299487, 0.038411] | -0.9062 | -0.6875 |
| transfer | dense | lag_mild | 1812 | mamps_shared_1812 | -0.148926 | [-0.319694, 0.009285] | -0.9062 | -0.6875 |
| transfer | dense | lag_mild | 1812 | mps_shared_1812 | -0.148926 | [-0.320312, 0.009119] | -0.9062 | -0.6875 |
| transfer | dense | lag_mild | 1822 | mamps_shared_1822 | -0.116048 | [-0.302572, 0.061849] | -0.9062 | -0.5625 |
| transfer | dense | lag_mild | 1822 | mps_shared_1822 | -0.111816 | [-0.298515, 0.062667] | -0.9062 | -0.5625 |
| transfer | dense | lag_strong | 1802 | mamps_shared_1802 | -2.264160 | [-4.582515, -0.102991] | -0.8125 | -0.4375 |
| transfer | dense | lag_strong | 1802 | mps_shared_1802 | -2.263509 | [-4.581848, -0.102340] | -0.8125 | -0.4375 |
| transfer | dense | lag_strong | 1812 | mamps_shared_1812 | -1.851725 | [-3.820508, 0.103528] | -0.7500 | -0.5000 |
| transfer | dense | lag_strong | 1812 | mps_shared_1812 | -1.855306 | [-3.825777, 0.103206] | -0.7500 | -0.5000 |
| transfer | dense | lag_strong | 1822 | mamps_shared_1822 | -1.763509 | [-4.138057, 0.562878] | -0.7812 | -0.4375 |
| transfer | dense | lag_strong | 1822 | mps_shared_1822 | -1.752767 | [-4.127681, 0.573812] | -0.7812 | -0.4688 |
| transfer | slow_yield | nominal | 1802 | mamps_shared_1802 | 0.890299 | [0.416943, 1.515340] | -0.9375 | -0.4688 |
| transfer | slow_yield | nominal | 1802 | mps_shared_1802 | 0.863607 | [0.397461, 1.533870] | -0.9375 | -0.4375 |
| transfer | slow_yield | nominal | 1812 | mamps_shared_1812 | 0.444661 | [0.300114, 0.589193] | -0.9375 | -0.4375 |
| transfer | slow_yield | nominal | 1812 | mps_shared_1812 | 0.459310 | [0.307292, 0.615234] | -0.9375 | -0.4375 |
| transfer | slow_yield | nominal | 1822 | mamps_shared_1822 | 0.782552 | [0.605143, 0.980827] | -0.9062 | -0.6667 |
| transfer | slow_yield | nominal | 1822 | mps_shared_1822 | 0.755208 | [0.578125, 0.961930] | -0.9062 | -0.6667 |
| transfer | slow_yield | gain_drag | 1802 | mamps_shared_1802 | 0.681315 | [0.337858, 1.090519] | -0.9688 | -0.9688 |
| transfer | slow_yield | gain_drag | 1802 | mps_shared_1802 | 0.763021 | [0.301099, 1.458993] | -0.9688 | -0.9688 |
| transfer | slow_yield | gain_drag | 1812 | mamps_shared_1812 | 0.435547 | [0.249349, 0.614583] | -0.9062 | -0.9688 |
| transfer | slow_yield | gain_drag | 1812 | mps_shared_1812 | 0.438802 | [0.267578, 0.600911] | -0.9062 | -0.9375 |
| transfer | slow_yield | gain_drag | 1822 | mamps_shared_1822 | 0.749349 | [0.503906, 1.098991] | -0.9062 | -1.0000 |
| transfer | slow_yield | gain_drag | 1822 | mps_shared_1822 | 0.785156 | [0.544271, 1.076823] | -0.9062 | -1.0000 |
| transfer | slow_yield | lag_mild | 1802 | mamps_shared_1802 | 0.834798 | [0.483232, 1.268901] | -0.9062 | -0.9688 |
| transfer | slow_yield | lag_mild | 1802 | mps_shared_1802 | 0.623861 | [0.404749, 0.849292] | -0.9062 | -0.9688 |
| transfer | slow_yield | lag_mild | 1812 | mamps_shared_1812 | 0.404460 | [0.242834, 0.562036] | -0.9375 | -0.9375 |
| transfer | slow_yield | lag_mild | 1812 | mps_shared_1812 | 0.417806 | [0.248368, 0.592611] | -0.9375 | -0.9375 |
| transfer | slow_yield | lag_mild | 1822 | mamps_shared_1822 | 0.745605 | [0.576172, 0.933919] | -0.9375 | -1.0000 |
| transfer | slow_yield | lag_mild | 1822 | mps_shared_1822 | 0.737142 | [0.551917, 0.945492] | -0.9375 | -1.0000 |
| transfer | slow_yield | lag_strong | 1802 | mamps_shared_1802 | 1.514974 | [-0.680013, 3.695435] | -0.9688 | -0.7812 |
| transfer | slow_yield | lag_strong | 1802 | mps_shared_1802 | 0.159831 | [-1.462996, 1.587980] | -0.9688 | -0.8125 |
| transfer | slow_yield | lag_strong | 1812 | mamps_shared_1812 | 0.971029 | [-1.181006, 3.149430] | -0.9375 | -0.8125 |
| transfer | slow_yield | lag_strong | 1812 | mps_shared_1812 | 0.569661 | [-1.063574, 2.236727] | -0.9375 | -0.8125 |
| transfer | slow_yield | lag_strong | 1822 | mamps_shared_1822 | 4.097005 | [1.573535, 6.618140] | -0.9375 | -0.9375 |
| transfer | slow_yield | lag_strong | 1822 | mps_shared_1822 | 3.876953 | [1.397420, 6.370841] | -0.9375 | -0.9375 |
| transfer | asymmetric_speed | nominal | 1802 | mamps_shared_1802 | 0.150065 | [0.052409, 0.298177] | -0.7188 | -0.8276 |
| transfer | asymmetric_speed | nominal | 1802 | mps_shared_1802 | 0.140299 | [0.042961, 0.291667] | -0.7188 | -0.8276 |
| transfer | asymmetric_speed | nominal | 1812 | mamps_shared_1812 | 0.076823 | [-0.035156, 0.238281] | -0.7188 | -0.5000 |
| transfer | asymmetric_speed | nominal | 1812 | mps_shared_1812 | 0.318034 | [-0.036784, 0.977555] | -0.7188 | -0.5000 |
| transfer | asymmetric_speed | nominal | 1822 | mamps_shared_1822 | 0.078939 | [-0.012695, 0.196940] | -0.7500 | -0.5714 |
| transfer | asymmetric_speed | nominal | 1822 | mps_shared_1822 | 0.078939 | [-0.012695, 0.196940] | -0.7500 | -0.5714 |
| transfer | asymmetric_speed | gain_drag | 1802 | mamps_shared_1802 | 0.139486 | [0.037431, 0.289559] | -0.7188 | -0.7931 |
| transfer | asymmetric_speed | gain_drag | 1802 | mps_shared_1802 | 0.124512 | [0.023763, 0.277507] | -0.7188 | -0.7931 |
| transfer | asymmetric_speed | gain_drag | 1812 | mamps_shared_1812 | 0.051921 | [-0.047856, 0.185238] | -0.7188 | -0.5000 |
| transfer | asymmetric_speed | gain_drag | 1812 | mps_shared_1812 | 0.046061 | [-0.048018, 0.168958] | -0.7188 | -0.5000 |
| transfer | asymmetric_speed | gain_drag | 1822 | mamps_shared_1822 | 0.064453 | [-0.017912, 0.171558] | -0.7188 | -0.5357 |
| transfer | asymmetric_speed | gain_drag | 1822 | mps_shared_1822 | 0.064453 | [-0.017912, 0.171558] | -0.7188 | -0.5357 |
| transfer | asymmetric_speed | lag_mild | 1802 | mamps_shared_1802 | 0.152018 | [0.040365, 0.325854] | -0.8125 | -0.8276 |
| transfer | asymmetric_speed | lag_mild | 1802 | mps_shared_1802 | 0.142253 | [0.032536, 0.316732] | -0.8125 | -0.8276 |
| transfer | asymmetric_speed | lag_mild | 1812 | mamps_shared_1812 | 0.083333 | [-0.027995, 0.243823] | -0.7188 | -0.5357 |
| transfer | asymmetric_speed | lag_mild | 1812 | mps_shared_1812 | 0.340169 | [-0.027026, 1.011759] | -0.7188 | -0.5357 |
| transfer | asymmetric_speed | lag_mild | 1822 | mamps_shared_1822 | 0.074544 | [-0.017904, 0.194019] | -0.8125 | -0.5714 |
| transfer | asymmetric_speed | lag_mild | 1822 | mps_shared_1822 | 0.074544 | [-0.017904, 0.194019] | -0.8125 | -0.5714 |
| transfer | asymmetric_speed | lag_strong | 1802 | mamps_shared_1802 | 0.321289 | [-0.173177, 1.233398] | -0.5625 | -0.4286 |
| transfer | asymmetric_speed | lag_strong | 1802 | mps_shared_1802 | 0.314779 | [-0.179036, 1.224951] | -0.5625 | -0.4286 |
| transfer | asymmetric_speed | lag_strong | 1812 | mamps_shared_1812 | 0.302897 | [-0.202490, 1.212732] | -0.6250 | -0.3214 |
| transfer | asymmetric_speed | lag_strong | 1812 | mps_shared_1812 | 0.301595 | [-0.202490, 1.208826] | -0.6250 | -0.3214 |
| transfer | asymmetric_speed | lag_strong | 1822 | mamps_shared_1822 | -0.076009 | [-0.225260, 0.099288] | -0.6250 | -0.3214 |
| transfer | asymmetric_speed | lag_strong | 1822 | mps_shared_1822 | -0.068197 | [-0.224626, 0.122888] | -0.6250 | -0.3214 |
| transfer | tight_headway | nominal | 1802 | mamps_shared_1802 | 0.039225 | [-0.098796, 0.225586] | -0.7500 | -0.5312 |
| transfer | tight_headway | nominal | 1802 | mps_shared_1802 | 0.009928 | [-0.086755, 0.114107] | -0.7500 | -0.5625 |
| transfer | tight_headway | nominal | 1812 | mamps_shared_1812 | -0.052734 | [-0.097819, -0.008301] | -0.7188 | -0.4688 |
| transfer | tight_headway | nominal | 1812 | mps_shared_1812 | -0.041667 | [-0.085942, 0.001139] | -0.7188 | -0.4688 |
| transfer | tight_headway | nominal | 1822 | mamps_shared_1822 | -0.016113 | [-0.084965, 0.074548] | -0.7500 | -0.4062 |
| transfer | tight_headway | nominal | 1822 | mps_shared_1822 | -0.028809 | [-0.090658, 0.057625] | -0.7500 | -0.3750 |
| transfer | tight_headway | gain_drag | 1802 | mamps_shared_1802 | 0.020345 | [-0.129883, 0.206559] | -0.7500 | -0.5625 |
| transfer | tight_headway | gain_drag | 1802 | mps_shared_1802 | -0.014160 | [-0.131510, 0.106934] | -0.7500 | -0.5625 |
| transfer | tight_headway | gain_drag | 1812 | mamps_shared_1812 | -0.053711 | [-0.141280, 0.022135] | -0.7500 | -0.4062 |
| transfer | tight_headway | gain_drag | 1812 | mps_shared_1812 | -0.053060 | [-0.146159, 0.026367] | -0.7500 | -0.3750 |
| transfer | tight_headway | gain_drag | 1822 | mamps_shared_1822 | -0.015951 | [-0.110522, 0.079102] | -0.7188 | -0.4688 |
| transfer | tight_headway | gain_drag | 1822 | mps_shared_1822 | -0.025391 | [-0.116211, 0.068848] | -0.7188 | -0.4375 |
| transfer | tight_headway | lag_mild | 1802 | mamps_shared_1802 | 0.193034 | [-0.086263, 0.672233] | -0.7500 | -0.5000 |
| transfer | tight_headway | lag_mild | 1802 | mps_shared_1802 | 0.020508 | [-0.076660, 0.126469] | -0.7500 | -0.5312 |
| transfer | tight_headway | lag_mild | 1812 | mamps_shared_1812 | -0.033691 | [-0.110352, 0.035482] | -0.8125 | -0.4688 |
| transfer | tight_headway | lag_mild | 1812 | mps_shared_1812 | -0.030436 | [-0.104659, 0.037923] | -0.8125 | -0.4688 |
| transfer | tight_headway | lag_mild | 1822 | mamps_shared_1822 | -0.018880 | [-0.107751, 0.076188] | -0.7500 | -0.3750 |
| transfer | tight_headway | lag_mild | 1822 | mps_shared_1822 | -0.031250 | [-0.116048, 0.062341] | -0.7500 | -0.4062 |
| transfer | tight_headway | lag_strong | 1802 | mamps_shared_1802 | -0.073242 | [-0.270207, 0.207381] | -0.6250 | -0.3542 |
| transfer | tight_headway | lag_strong | 1802 | mps_shared_1802 | -0.141602 | [-0.252612, -0.037427] | -0.6250 | -0.3542 |
| transfer | tight_headway | lag_strong | 1812 | mamps_shared_1812 | -0.667969 | [-1.693368, -0.111971] | -0.7188 | -0.4062 |
| transfer | tight_headway | lag_strong | 1812 | mps_shared_1812 | -0.660807 | [-1.674487, -0.115877] | -0.7188 | -0.4062 |
| transfer | tight_headway | lag_strong | 1822 | mamps_shared_1822 | -0.215820 | [-1.646493, 1.237020] | -0.6562 | -0.2528 |
| transfer | tight_headway | lag_strong | 1822 | mps_shared_1822 | -0.174154 | [-1.603540, 1.274113] | -0.6562 | -0.2216 |

## All 500 transfer cells

Each cell has 32 matched initials. Models and plants do not multiply the independent sample count. Baseline failures and physical error exceedances remain visible. No safety claim covers motion after an abort.

| Scene | Plant | Method | Gain % | Gain 95% interval (s) | Adopted / N | Harmful / adopted | Original adopted | Failed | Baseline failed | Error-exceeded |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|
| nominal | nominal | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| nominal | nominal | raw_1802 | 0.3678 | [-0.093115, 0.166016] | 32/32 | 16/32 | 32 | 0 | 0 | 0 |
| nominal | nominal | nominal_1802 | 1.1098 | [0.041650, 0.211589] | 19/32 | 1/19 | 19 | 0 | 0 | 0 |
| nominal | nominal | portfolio0_1802 | 0.3162 | [0.001302, 0.088542] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| nominal | nominal | repair_1802 | 0.5839 | [0.006836, 0.132812] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| nominal | nominal | phase05_1802 | 0.5597 | [0.004883, 0.131026] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| nominal | nominal | phase10_1802 | 0.5775 | [0.006510, 0.132161] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| nominal | nominal | raw_1812 | 0.4581 | [-0.055990, 0.174479] | 32/32 | 22/32 | 32 | 0 | 0 | 0 |
| nominal | nominal | nominal_1812 | 0.6840 | [0.032552, 0.118490] | 21/32 | 1/21 | 21 | 0 | 0 | 0 |
| nominal | nominal | portfolio0_1812 | 0.2516 | [0.000000, 0.070964] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| nominal | nominal | repair_1812 | 0.5565 | [0.004883, 0.130538] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| nominal | nominal | phase05_1812 | 0.5501 | [0.004395, 0.129883] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| nominal | nominal | phase10_1812 | 0.5565 | [0.004883, 0.130538] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| nominal | nominal | raw_1822 | -0.1097 | [-0.121419, 0.111328] | 32/32 | 19/32 | 32 | 0 | 0 | 0 |
| nominal | nominal | nominal_1822 | 0.9872 | [0.029948, 0.198568] | 15/32 | 0/15 | 15 | 0 | 0 | 0 |
| nominal | nominal | portfolio0_1822 | 0.4968 | [0.000651, 0.124349] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| nominal | nominal | repair_1822 | 0.5468 | [0.004557, 0.128906] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| nominal | nominal | phase05_1822 | 0.5339 | [0.003581, 0.128097] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| nominal | nominal | phase10_1822 | 0.5468 | [0.004557, 0.128906] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| nominal | nominal | mamps_shared_1802 | 0.5356 | [-0.071615, 0.181657] | 32/32 | 13/32 | 32 | 0 | 0 | 0 |
| nominal | nominal | mps_shared_1802 | 0.5420 | [-0.070964, 0.182292] | 32/32 | 13/32 | 32 | 0 | 0 | 0 |
| nominal | nominal | mamps_shared_1812 | 0.8388 | [-0.008480, 0.203776] | 32/32 | 20/32 | 32 | 0 | 0 | 0 |
| nominal | nominal | mps_shared_1812 | 0.8324 | [-0.009131, 0.203141] | 32/32 | 20/32 | 32 | 0 | 0 | 0 |
| nominal | nominal | mamps_shared_1822 | 0.3871 | [-0.069344, 0.157886] | 32/32 | 16/32 | 32 | 0 | 0 | 0 |
| nominal | nominal | mps_shared_1822 | 0.3226 | [-0.076823, 0.152352] | 32/32 | 16/32 | 32 | 0 | 0 | 0 |
| nominal | gain_drag | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| nominal | gain_drag | raw_1802 | 0.1861 | [-0.097005, 0.134766] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| nominal | gain_drag | nominal_1802 | 0.7767 | [0.015625, 0.177751] | 18/32 | 5/18 | 18 | 0 | 0 | 0 |
| nominal | gain_drag | portfolio0_1802 | 0.2375 | [0.001302, 0.065755] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| nominal | gain_drag | repair_1802 | 0.0714 | [0.002279, 0.013265] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| nominal | gain_drag | phase05_1802 | 0.0586 | [0.001546, 0.011477] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| nominal | gain_drag | phase10_1802 | 0.0698 | [0.002197, 0.013021] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| nominal | gain_drag | raw_1812 | 0.6034 | [-0.015641, 0.160173] | 32/32 | 18/32 | 32 | 0 | 0 | 0 |
| nominal | gain_drag | nominal_1812 | 0.7767 | [0.031901, 0.135417] | 19/32 | 3/19 | 19 | 0 | 0 | 0 |
| nominal | gain_drag | portfolio0_1812 | 0.1990 | [-0.006510, 0.065104] | 6/32 | 1/6 | 6 | 0 | 0 | 0 |
| nominal | gain_drag | repair_1812 | 0.1187 | [0.003743, 0.024089] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| nominal | gain_drag | phase05_1812 | 0.0530 | [0.001302, 0.010258] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| nominal | gain_drag | phase10_1812 | 0.0658 | [0.002604, 0.011719] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| nominal | gain_drag | raw_1822 | 0.0032 | [-0.106779, 0.108724] | 32/32 | 21/32 | 32 | 0 | 0 | 0 |
| nominal | gain_drag | nominal_1822 | 0.7574 | [0.028646, 0.145833] | 17/32 | 2/17 | 17 | 0 | 0 | 0 |
| nominal | gain_drag | portfolio0_1822 | 0.4172 | [0.000651, 0.124349] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| nominal | gain_drag | repair_1822 | 0.0963 | [0.001302, 0.022135] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| nominal | gain_drag | phase05_1822 | 0.0385 | [0.000000, 0.009115] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| nominal | gain_drag | phase10_1822 | 0.0449 | [0.000651, 0.009766] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| nominal | gain_drag | mamps_shared_1802 | 0.3466 | [-0.080078, 0.147152] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| nominal | gain_drag | mps_shared_1802 | 0.3530 | [-0.079427, 0.147786] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| nominal | gain_drag | mamps_shared_1812 | 0.8986 | [0.015625, 0.194661] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| nominal | gain_drag | mps_shared_1812 | 0.8986 | [0.015609, 0.195312] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| nominal | gain_drag | mamps_shared_1822 | 0.2535 | [-0.068359, 0.124023] | 32/32 | 17/32 | 32 | 0 | 0 | 0 |
| nominal | gain_drag | mps_shared_1822 | 0.2054 | [-0.072282, 0.119141] | 32/32 | 17/32 | 32 | 0 | 0 | 0 |
| nominal | lag_mild | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| nominal | lag_mild | raw_1802 | -0.0583 | [-0.117204, 0.085303] | 32/32 | 16/32 | 32 | 0 | 0 | 0 |
| nominal | lag_mild | nominal_1802 | 0.5116 | [0.017578, 0.098958] | 19/32 | 2/19 | 19 | 0 | 0 | 0 |
| nominal | lag_mild | portfolio0_1802 | 0.0389 | [0.000651, 0.007813] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| nominal | lag_mild | repair_1802 | 0.0729 | [0.002604, 0.012858] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| nominal | lag_mild | phase05_1802 | 0.0534 | [0.001302, 0.010254] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| nominal | lag_mild | phase10_1802 | 0.0648 | [0.002279, 0.012044] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| nominal | lag_mild | raw_1812 | 0.1295 | [-0.057943, 0.098307] | 32/32 | 22/32 | 32 | 0 | 0 | 0 |
| nominal | lag_mild | nominal_1812 | 0.4598 | [0.026042, 0.067708] | 19/32 | 1/19 | 19 | 0 | 0 | 0 |
| nominal | lag_mild | portfolio0_1812 | 0.1101 | [0.001953, 0.024089] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| nominal | lag_mild | repair_1812 | 0.0680 | [0.002600, 0.012044] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| nominal | lag_mild | phase05_1812 | 0.0567 | [0.001302, 0.010905] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| nominal | lag_mild | phase10_1812 | 0.0631 | [0.002116, 0.011719] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| nominal | lag_mild | raw_1822 | -0.3691 | [-0.125334, 0.049487] | 32/32 | 20/32 | 32 | 0 | 0 | 0 |
| nominal | lag_mild | nominal_1822 | 0.5504 | [0.023438, 0.102214] | 15/32 | 0/15 | 15 | 0 | 0 | 0 |
| nominal | lag_mild | portfolio0_1822 | 0.2008 | [0.000000, 0.059896] | 2/32 | 0/2 | 2 | 0 | 0 | 0 |
| nominal | lag_mild | repair_1822 | 0.0389 | [0.000651, 0.008138] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| nominal | lag_mild | phase05_1822 | 0.0324 | [0.000000, 0.007487] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| nominal | lag_mild | phase10_1822 | 0.0389 | [0.000651, 0.008138] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| nominal | lag_mild | mamps_shared_1802 | 0.0907 | [-0.098307, 0.095052] | 32/32 | 14/32 | 32 | 0 | 0 | 0 |
| nominal | lag_mild | mps_shared_1802 | 0.0971 | [-0.097656, 0.095068] | 32/32 | 14/32 | 32 | 0 | 0 | 0 |
| nominal | lag_mild | mamps_shared_1812 | 0.5310 | [-0.003906, 0.131510] | 32/32 | 20/32 | 32 | 0 | 0 | 0 |
| nominal | lag_mild | mps_shared_1812 | 0.5375 | [-0.001953, 0.131510] | 32/32 | 20/32 | 32 | 0 | 0 | 0 |
| nominal | lag_mild | mamps_shared_1822 | -0.0712 | [-0.095060, 0.078125] | 32/32 | 17/32 | 32 | 0 | 0 | 0 |
| nominal | lag_mild | mps_shared_1822 | -0.1425 | [-0.102865, 0.071948] | 32/32 | 17/32 | 32 | 0 | 0 | 0 |
| nominal | lag_strong | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 32 |
| nominal | lag_strong | raw_1802 | 1.1948 | [0.014958, 0.222005] | 32/32 | 15/32 | 32 | 0 | 0 | 32 |
| nominal | lag_strong | nominal_1802 | 0.7406 | [0.033203, 0.128255] | 22/32 | 4/22 | 22 | 0 | 0 | 32 |
| nominal | lag_strong | portfolio0_1802 | 0.1680 | [0.000651, 0.047526] | 4/32 | 0/4 | 4 | 0 | 0 | 32 |
| nominal | lag_strong | repair_1802 | 0.1462 | [-0.003581, 0.036621] | 10/32 | 1/10 | 10 | 0 | 0 | 32 |
| nominal | lag_strong | phase05_1802 | 0.0187 | [-0.014327, 0.018066] | 7/32 | 2/7 | 7 | 0 | 0 | 32 |
| nominal | lag_strong | phase10_1802 | 0.0342 | [-0.013021, 0.020024] | 9/32 | 2/9 | 9 | 0 | 0 | 32 |
| nominal | lag_strong | raw_1812 | 1.6958 | [0.083976, 0.284180] | 32/32 | 12/32 | 32 | 0 | 0 | 32 |
| nominal | lag_strong | nominal_1812 | 0.8277 | [0.027344, 0.158854] | 20/32 | 6/20 | 20 | 0 | 0 | 32 |
| nominal | lag_strong | portfolio0_1812 | 0.1058 | [0.000651, 0.027995] | 4/32 | 0/4 | 4 | 0 | 0 | 32 |
| nominal | lag_strong | repair_1812 | 0.2598 | [-0.000651, 0.062992] | 10/32 | 1/10 | 10 | 0 | 0 | 32 |
| nominal | lag_strong | phase05_1812 | 0.0233 | [-0.014323, 0.018555] | 8/32 | 2/8 | 8 | 0 | 0 | 32 |
| nominal | lag_strong | phase10_1812 | 0.1540 | [-0.009603, 0.049805] | 10/32 | 2/10 | 10 | 0 | 0 | 32 |
| nominal | lag_strong | raw_1822 | 1.4251 | [0.050130, 0.244792] | 31/32 | 16/31 | 32 | 0 | 0 | 32 |
| nominal | lag_strong | nominal_1822 | 0.6099 | [0.021484, 0.114583] | 21/32 | 7/21 | 21 | 0 | 0 | 32 |
| nominal | lag_strong | portfolio0_1822 | 0.0871 | [0.000000, 0.026042] | 2/32 | 0/2 | 2 | 0 | 0 | 32 |
| nominal | lag_strong | repair_1822 | 0.2458 | [0.006836, 0.049972] | 11/32 | 0/11 | 11 | 0 | 0 | 32 |
| nominal | lag_strong | phase05_1822 | 0.1151 | [-0.003743, 0.030762] | 8/32 | 1/8 | 8 | 0 | 0 | 32 |
| nominal | lag_strong | phase10_1822 | 0.2458 | [0.000163, 0.061039] | 10/32 | 1/10 | 10 | 0 | 0 | 32 |
| nominal | lag_strong | mamps_shared_1802 | 1.4002 | [0.040348, 0.242187] | 32/32 | 12/32 | 32 | 0 | 0 | 32 |
| nominal | lag_strong | mps_shared_1802 | 1.3815 | [0.038411, 0.238949] | 32/32 | 12/32 | 32 | 0 | 0 | 32 |
| nominal | lag_strong | mamps_shared_1812 | 1.9665 | [0.117839, 0.311849] | 32/32 | 10/32 | 32 | 0 | 0 | 32 |
| nominal | lag_strong | mps_shared_1812 | 1.9665 | [0.117839, 0.311849] | 32/32 | 10/32 | 32 | 0 | 0 | 32 |
| nominal | lag_strong | mamps_shared_1822 | 1.8141 | [0.099927, 0.289062] | 31/32 | 14/31 | 31 | 0 | 0 | 32 |
| nominal | lag_strong | mps_shared_1822 | 1.7767 | [0.096672, 0.285173] | 31/32 | 14/31 | 31 | 0 | 0 | 32 |
| dense | nominal | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| dense | nominal | raw_1802 | 0.5730 | [-0.098958, 0.213867] | 32/32 | 26/32 | 32 | 0 | 0 | 0 |
| dense | nominal | nominal_1802 | 1.7912 | [0.089193, 0.264974] | 23/32 | 4/23 | 23 | 0 | 0 | 0 |
| dense | nominal | portfolio0_1802 | 0.9196 | [0.025391, 0.163411] | 7/32 | 1/7 | 7 | 0 | 0 | 0 |
| dense | nominal | repair_1802 | 0.7257 | [0.013021, 0.142586] | 6/32 | 1/6 | 6 | 0 | 0 | 0 |
| dense | nominal | phase05_1802 | 0.7309 | [0.013021, 0.143559] | 6/32 | 1/6 | 6 | 0 | 0 | 0 |
| dense | nominal | phase10_1802 | 0.7309 | [0.013021, 0.143559] | 6/32 | 1/6 | 6 | 0 | 0 | 0 |
| dense | nominal | raw_1812 | 1.0294 | [-0.043302, 0.247729] | 32/32 | 25/32 | 32 | 0 | 0 | 0 |
| dense | nominal | nominal_1812 | 1.4274 | [0.069010, 0.214860] | 21/32 | 3/21 | 21 | 0 | 0 | 0 |
| dense | nominal | portfolio0_1812 | 0.5627 | [0.011068, 0.108724] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| dense | nominal | repair_1812 | 0.3586 | [0.000163, 0.097823] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| dense | nominal | phase05_1812 | 0.2556 | [0.000000, 0.072266] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| dense | nominal | phase10_1812 | 0.3380 | [0.000163, 0.091964] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| dense | nominal | raw_1822 | 0.7823 | [-0.101245, 0.249357] | 32/32 | 24/32 | 32 | 0 | 0 | 0 |
| dense | nominal | nominal_1822 | 1.9868 | [0.102865, 0.286133] | 25/32 | 2/25 | 25 | 0 | 0 | 0 |
| dense | nominal | portfolio0_1822 | 0.7446 | [0.009098, 0.150724] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| dense | nominal | repair_1822 | 0.4272 | [0.000163, 0.104818] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| dense | nominal | phase05_1822 | 0.4186 | [0.000000, 0.105143] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| dense | nominal | phase10_1822 | 0.4324 | [0.000163, 0.106283] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| dense | nominal | mamps_shared_1802 | 1.1907 | [-0.044287, 0.273462] | 32/32 | 23/32 | 32 | 0 | 0 | 0 |
| dense | nominal | mps_shared_1802 | 1.1632 | [-0.046549, 0.270207] | 32/32 | 23/32 | 32 | 0 | 0 | 0 |
| dense | nominal | mamps_shared_1812 | 1.4583 | [-0.007821, 0.294596] | 32/32 | 25/32 | 32 | 0 | 0 | 0 |
| dense | nominal | mps_shared_1812 | 1.4652 | [-0.006510, 0.294295] | 32/32 | 25/32 | 32 | 0 | 0 | 0 |
| dense | nominal | mamps_shared_1822 | 1.2284 | [-0.065430, 0.295589] | 32/32 | 22/32 | 32 | 0 | 0 | 0 |
| dense | nominal | mps_shared_1822 | 1.1667 | [-0.070321, 0.289388] | 32/32 | 22/32 | 32 | 0 | 0 | 0 |
| dense | gain_drag | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| dense | gain_drag | raw_1802 | 1.4634 | [-0.018245, 0.306006] | 32/32 | 22/32 | 32 | 0 | 0 | 0 |
| dense | gain_drag | nominal_1802 | 1.6812 | [0.066406, 0.270182] | 21/32 | 6/21 | 21 | 0 | 0 | 0 |
| dense | gain_drag | portfolio0_1802 | 0.8780 | [0.001953, 0.181641] | 8/32 | 1/8 | 8 | 0 | 0 | 0 |
| dense | gain_drag | repair_1802 | 0.9801 | [0.024089, 0.182292] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| dense | gain_drag | phase05_1802 | 0.9665 | [0.022786, 0.181006] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| dense | gain_drag | phase10_1802 | 0.9665 | [0.022786, 0.181006] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| dense | gain_drag | raw_1812 | 1.5689 | [-0.004891, 0.308927] | 32/32 | 21/32 | 32 | 0 | 0 | 0 |
| dense | gain_drag | nominal_1812 | 1.5587 | [0.073568, 0.239583] | 18/32 | 5/18 | 18 | 0 | 0 | 0 |
| dense | gain_drag | portfolio0_1812 | 0.8168 | [0.015625, 0.153646] | 8/32 | 1/8 | 8 | 0 | 0 | 0 |
| dense | gain_drag | repair_1812 | 0.8032 | [0.014323, 0.158854] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| dense | gain_drag | phase05_1812 | 0.7895 | [0.013672, 0.157552] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| dense | gain_drag | phase10_1812 | 0.8032 | [0.014323, 0.158854] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| dense | gain_drag | raw_1822 | -1.8445 | [-0.907935, 0.275716] | 32/32 | 21/32 | 32 | 1 | 0 | 0 |
| dense | gain_drag | nominal_1822 | 2.0351 | [0.106104, 0.297526] | 26/32 | 4/26 | 26 | 0 | 0 | 0 |
| dense | gain_drag | portfolio0_1822 | 1.0278 | [0.031901, 0.181641] | 9/32 | 1/9 | 9 | 0 | 0 | 0 |
| dense | gain_drag | repair_1822 | 0.6075 | [0.011877, 0.122559] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| dense | gain_drag | phase05_1822 | 0.5922 | [0.009766, 0.121745] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| dense | gain_drag | phase10_1822 | 0.6058 | [0.011719, 0.122396] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| dense | gain_drag | mamps_shared_1802 | 1.6267 | [-0.007829, 0.324870] | 32/32 | 20/32 | 32 | 0 | 0 | 0 |
| dense | gain_drag | mps_shared_1802 | 1.6608 | [-0.003923, 0.327474] | 32/32 | 20/32 | 32 | 0 | 0 | 0 |
| dense | gain_drag | mamps_shared_1812 | 1.7118 | [0.007479, 0.324227] | 32/32 | 21/32 | 32 | 0 | 0 | 0 |
| dense | gain_drag | mps_shared_1812 | 1.7152 | [0.007813, 0.324235] | 32/32 | 21/32 | 32 | 0 | 0 | 0 |
| dense | gain_drag | mamps_shared_1822 | 1.3783 | [-0.063485, 0.317057] | 32/32 | 17/32 | 32 | 0 | 0 | 0 |
| dense | gain_drag | mps_shared_1822 | 1.5280 | [-0.045898, 0.328133] | 32/32 | 17/32 | 32 | 0 | 0 | 0 |
| dense | lag_mild | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| dense | lag_mild | raw_1802 | 1.0766 | [-0.062183, 0.276709] | 32/32 | 24/32 | 32 | 0 | 0 | 0 |
| dense | lag_mild | nominal_1802 | 1.9003 | [0.078776, 0.298193] | 22/32 | 5/22 | 22 | 0 | 0 | 0 |
| dense | lag_mild | portfolio0_1802 | 0.9638 | [0.019515, 0.186198] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| dense | lag_mild | repair_1802 | 0.3076 | [0.000651, 0.068359] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| dense | lag_mild | phase05_1802 | 0.3008 | [0.000000, 0.068359] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| dense | lag_mild | phase10_1802 | 0.3076 | [0.000651, 0.068359] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| dense | lag_mild | raw_1812 | 1.3910 | [-0.024764, 0.299495] | 32/32 | 22/32 | 32 | 0 | 0 | 0 |
| dense | lag_mild | nominal_1812 | 1.2577 | [0.050781, 0.207682] | 17/32 | 1/17 | 17 | 0 | 0 | 0 |
| dense | lag_mild | portfolio0_1812 | 0.4648 | [0.001953, 0.111328] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| dense | lag_mild | repair_1812 | 0.2136 | [0.002116, 0.049154] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| dense | lag_mild | phase05_1812 | 0.1794 | [0.000000, 0.045573] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| dense | lag_mild | phase10_1812 | 0.1931 | [0.000814, 0.046391] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| dense | lag_mild | raw_1822 | 1.1005 | [-0.069686, 0.286466] | 32/32 | 21/32 | 32 | 0 | 0 | 0 |
| dense | lag_mild | nominal_1822 | 1.9310 | [0.096997, 0.285164] | 23/32 | 2/23 | 23 | 0 | 0 | 0 |
| dense | lag_mild | portfolio0_1822 | 0.8373 | [0.018229, 0.163086] | 10/32 | 1/10 | 10 | 0 | 0 | 0 |
| dense | lag_mild | repair_1822 | 0.2341 | [0.000814, 0.058110] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| dense | lag_mild | phase05_1822 | 0.2204 | [0.000000, 0.057292] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| dense | lag_mild | phase10_1822 | 0.2341 | [0.000814, 0.058110] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| dense | lag_mild | mamps_shared_1802 | 1.6303 | [-0.016626, 0.329427] | 32/32 | 22/32 | 32 | 0 | 0 | 0 |
| dense | lag_mild | mps_shared_1802 | 1.6303 | [-0.016626, 0.329427] | 32/32 | 22/32 | 32 | 0 | 0 | 0 |
| dense | lag_mild | mamps_shared_1812 | 1.7431 | [0.007804, 0.333008] | 32/32 | 22/32 | 32 | 0 | 0 | 0 |
| dense | lag_mild | mps_shared_1812 | 1.7431 | [0.008138, 0.333659] | 32/32 | 22/32 | 32 | 0 | 0 | 0 |
| dense | lag_mild | mamps_shared_1822 | 1.4389 | [-0.043294, 0.321289] | 32/32 | 18/32 | 32 | 0 | 0 | 0 |
| dense | lag_mild | mps_shared_1822 | 1.3944 | [-0.046558, 0.317065] | 32/32 | 18/32 | 32 | 0 | 0 | 0 |
| dense | lag_strong | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 5 | 5 | 32 |
| dense | lag_strong | raw_1802 | 22.2751 | [0.983203, 4.746851] | 32/32 | 16/32 | 32 | 0 | 5 | 32 |
| dense | lag_strong | nominal_1802 | 9.1380 | [0.162435, 2.543351] | 26/32 | 6/26 | 26 | 3 | 5 | 32 |
| dense | lag_strong | portfolio0_1802 | 4.1137 | [0.000651, 1.475586] | 6/32 | 0/6 | 6 | 4 | 5 | 32 |
| dense | lag_strong | repair_1802 | 4.4636 | [0.006181, 1.592293] | 10/32 | 0/10 | 10 | 4 | 5 | 32 |
| dense | lag_strong | phase05_1802 | 0.5113 | [0.001139, 0.186686] | 6/32 | 0/6 | 6 | 5 | 5 | 32 |
| dense | lag_strong | phase10_1802 | 0.5113 | [0.001139, 0.186686] | 6/32 | 0/6 | 6 | 5 | 5 | 32 |
| dense | lag_strong | raw_1812 | 22.0071 | [0.955957, 4.707373] | 32/32 | 17/32 | 32 | 0 | 5 | 32 |
| dense | lag_strong | nominal_1812 | 5.5057 | [-0.761735, 2.471785] | 22/32 | 10/22 | 22 | 4 | 5 | 32 |
| dense | lag_strong | portfolio0_1812 | 7.3192 | [0.000651, 2.281909] | 6/32 | 0/6 | 6 | 3 | 5 | 32 |
| dense | lag_strong | repair_1812 | 3.6752 | [-0.010095, 1.387048] | 12/32 | 2/12 | 12 | 4 | 5 | 32 |
| dense | lag_strong | phase05_1812 | 3.7273 | [0.003088, 1.388676] | 8/32 | 0/8 | 8 | 4 | 5 | 32 |
| dense | lag_strong | phase10_1812 | 3.6648 | [-0.011230, 1.385742] | 10/32 | 2/10 | 10 | 4 | 5 | 32 |
| dense | lag_strong | raw_1822 | 18.0106 | [0.304028, 4.550464] | 32/32 | 14/32 | 32 | 1 | 5 | 32 |
| dense | lag_strong | nominal_1822 | 6.4580 | [-0.731877, 2.578231] | 26/32 | 7/26 | 26 | 4 | 5 | 32 |
| dense | lag_strong | portfolio0_1822 | 4.7980 | [0.018864, 1.647803] | 9/32 | 1/9 | 9 | 4 | 5 | 32 |
| dense | lag_strong | repair_1822 | 0.8053 | [0.008301, 0.252934] | 13/32 | 1/13 | 13 | 5 | 5 | 32 |
| dense | lag_strong | phase05_1822 | 0.5737 | [0.001790, 0.207682] | 7/32 | 0/7 | 7 | 5 | 5 | 32 |
| dense | lag_strong | phase10_1822 | 0.5035 | [-0.015299, 0.204268] | 9/32 | 2/9 | 9 | 5 | 5 | 32 |
| dense | lag_strong | mamps_shared_1802 | 18.6090 | [0.167896, 4.647559] | 32/32 | 14/32 | 32 | 1 | 5 | 32 |
| dense | lag_strong | mps_shared_1802 | 18.6038 | [0.167879, 4.647526] | 32/32 | 14/32 | 32 | 1 | 5 | 32 |
| dense | lag_strong | mamps_shared_1812 | 18.5283 | [0.287044, 4.584318] | 32/32 | 16/32 | 32 | 1 | 5 | 32 |
| dense | lag_strong | mps_shared_1812 | 18.5570 | [0.289665, 4.584717] | 32/32 | 16/32 | 32 | 1 | 5 | 32 |
| dense | lag_strong | mamps_shared_1822 | 14.6697 | [-0.410498, 4.174251] | 32/32 | 14/32 | 32 | 2 | 5 | 32 |
| dense | lag_strong | mps_shared_1822 | 14.5838 | [-0.423511, 4.164095] | 32/32 | 15/32 | 32 | 2 | 5 | 32 |
| slow_yield | nominal | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| slow_yield | nominal | raw_1802 | -8.3434 | [-1.159538, -0.605452] | 32/32 | 31/32 | 32 | 0 | 0 | 0 |
| slow_yield | nominal | nominal_1802 | 0.2655 | [0.009766, 0.051432] | 15/32 | 1/15 | 15 | 0 | 0 | 0 |
| slow_yield | nominal | portfolio0_1802 | 0.0370 | [0.000000, 0.009115] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| slow_yield | nominal | repair_1802 | -0.0401 | [-0.016602, 0.004883] | 5/32 | 2/5 | 5 | 0 | 0 | 0 |
| slow_yield | nominal | phase05_1802 | -0.0401 | [-0.013672, 0.000977] | 2/32 | 1/2 | 2 | 0 | 0 | 0 |
| slow_yield | nominal | phase10_1802 | -0.0401 | [-0.013672, 0.000977] | 2/32 | 1/2 | 2 | 0 | 0 | 0 |
| slow_yield | nominal | raw_1812 | -6.8401 | [-1.127336, -0.465169] | 32/32 | 31/32 | 32 | 0 | 0 | 0 |
| slow_yield | nominal | nominal_1812 | 0.4661 | [0.025391, 0.076497] | 18/32 | 2/18 | 18 | 0 | 0 | 0 |
| slow_yield | nominal | portfolio0_1812 | 0.3642 | [0.000651, 0.102214] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| slow_yield | nominal | repair_1812 | 0.0216 | [-0.009766, 0.012044] | 6/32 | 1/6 | 6 | 0 | 0 | 0 |
| slow_yield | nominal | phase05_1812 | -0.0401 | [-0.013672, 0.000977] | 2/32 | 1/2 | 2 | 0 | 0 | 0 |
| slow_yield | nominal | phase10_1812 | -0.0401 | [-0.013672, 0.000977] | 2/32 | 1/2 | 2 | 0 | 0 | 0 |
| slow_yield | nominal | raw_1822 | -8.3434 | [-1.160807, -0.647135] | 32/32 | 32/32 | 32 | 0 | 0 | 0 |
| slow_yield | nominal | nominal_1822 | 0.5741 | [0.011719, 0.126969] | 18/32 | 1/18 | 18 | 0 | 0 | 0 |
| slow_yield | nominal | portfolio0_1822 | 0.3828 | [0.002604, 0.103532] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| slow_yield | nominal | repair_1822 | 0.0586 | [-0.012370, 0.029297] | 7/32 | 2/7 | 7 | 0 | 0 | 0 |
| slow_yield | nominal | phase05_1822 | 0.0463 | [-0.013346, 0.027669] | 3/32 | 1/3 | 3 | 0 | 0 | 0 |
| slow_yield | nominal | phase10_1822 | 0.0463 | [-0.013346, 0.027669] | 3/32 | 1/3 | 3 | 0 | 0 | 0 |
| slow_yield | nominal | mamps_shared_1802 | -8.4823 | [-1.520199, -0.421216] | 32/32 | 31/32 | 32 | 1 | 0 | 0 |
| slow_yield | nominal | mps_shared_1802 | -8.2292 | [-1.538411, -0.403638] | 32/32 | 30/32 | 32 | 1 | 0 | 0 |
| slow_yield | nominal | mamps_shared_1812 | -4.2566 | [-0.590495, -0.306641] | 32/32 | 30/32 | 32 | 0 | 0 | 0 |
| slow_yield | nominal | mps_shared_1812 | -4.3955 | [-0.617187, -0.311849] | 32/32 | 30/32 | 32 | 0 | 0 | 0 |
| slow_yield | nominal | mamps_shared_1822 | -7.3741 | [-0.967456, -0.606771] | 32/32 | 32/32 | 32 | 0 | 0 | 0 |
| slow_yield | nominal | mps_shared_1822 | -7.1149 | [-0.954435, -0.577783] | 32/32 | 32/32 | 32 | 0 | 0 | 0 |
| slow_yield | gain_drag | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| slow_yield | gain_drag | raw_1802 | -8.3293 | [-1.195329, -0.578776] | 32/32 | 31/32 | 32 | 0 | 0 | 0 |
| slow_yield | gain_drag | nominal_1802 | 0.2498 | [0.001953, 0.059245] | 13/32 | 2/13 | 13 | 0 | 0 | 0 |
| slow_yield | gain_drag | portfolio0_1802 | 0.0122 | [0.000000, 0.003906] | 1/32 | 0/1 | 1 | 0 | 0 | 0 |
| slow_yield | gain_drag | repair_1802 | 0.0244 | [0.000000, 0.007812] | 1/32 | 0/1 | 1 | 0 | 0 | 0 |
| slow_yield | gain_drag | phase05_1802 | 0.0061 | [0.000000, 0.001953] | 1/32 | 0/1 | 1 | 0 | 0 | 0 |
| slow_yield | gain_drag | phase10_1802 | 0.0061 | [0.000000, 0.001953] | 1/32 | 0/1 | 1 | 0 | 0 | 0 |
| slow_yield | gain_drag | raw_1812 | -8.3171 | [-1.677384, -0.414372] | 32/32 | 31/32 | 32 | 1 | 0 | 0 |
| slow_yield | gain_drag | nominal_1812 | 0.4814 | [0.009440, 0.102873] | 18/32 | 5/18 | 18 | 0 | 0 | 0 |
| slow_yield | gain_drag | portfolio0_1812 | 0.2315 | [-0.001302, 0.073568] | 4/32 | 1/4 | 4 | 0 | 0 | 0 |
| slow_yield | gain_drag | repair_1812 | 0.2315 | [0.001302, 0.067057] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| slow_yield | gain_drag | phase05_1812 | 0.2072 | [0.000000, 0.063151] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| slow_yield | gain_drag | phase10_1812 | 0.2072 | [0.000000, 0.063151] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| slow_yield | gain_drag | raw_1822 | -11.1717 | [-1.890348, -0.710938] | 32/32 | 32/32 | 32 | 1 | 0 | 0 |
| slow_yield | gain_drag | nominal_1822 | 0.4204 | [-0.003255, 0.105485] | 14/32 | 4/14 | 14 | 0 | 0 | 0 |
| slow_yield | gain_drag | portfolio0_1822 | 0.4448 | [0.000000, 0.117187] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| slow_yield | gain_drag | repair_1822 | 0.2681 | [0.001302, 0.078125] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| slow_yield | gain_drag | phase05_1822 | 0.2376 | [0.000000, 0.072917] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| slow_yield | gain_drag | phase10_1822 | 0.2376 | [0.000000, 0.072917] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| slow_yield | gain_drag | mamps_shared_1802 | -6.3703 | [-1.089876, -0.336572] | 32/32 | 31/32 | 32 | 0 | 0 | 0 |
| slow_yield | gain_drag | mps_shared_1802 | -7.1350 | [-1.458667, -0.300448] | 32/32 | 31/32 | 32 | 1 | 0 | 0 |
| slow_yield | gain_drag | mamps_shared_1812 | -3.8691 | [-0.592773, -0.228825] | 32/32 | 31/32 | 32 | 0 | 0 | 0 |
| slow_yield | gain_drag | mps_shared_1812 | -3.8996 | [-0.581380, -0.245426] | 32/32 | 30/32 | 32 | 0 | 0 | 0 |
| slow_yield | gain_drag | mamps_shared_1822 | -6.7755 | [-1.068359, -0.486312] | 32/32 | 32/32 | 32 | 0 | 0 | 0 |
| slow_yield | gain_drag | mps_shared_1822 | -7.1107 | [-1.050146, -0.522786] | 32/32 | 32/32 | 32 | 0 | 0 | 0 |
| slow_yield | lag_mild | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| slow_yield | lag_mild | raw_1802 | -8.2478 | [-1.155599, -0.596663] | 32/32 | 31/32 | 32 | 0 | 0 | 0 |
| slow_yield | lag_mild | nominal_1802 | 0.2889 | [0.008789, 0.055990] | 13/32 | 2/13 | 13 | 0 | 0 | 0 |
| slow_yield | lag_mild | portfolio0_1802 | 0.3411 | [0.000651, 0.103516] | 4/32 | 1/4 | 4 | 0 | 0 | 0 |
| slow_yield | lag_mild | repair_1802 | 0.3288 | [-0.002938, 0.101563] | 6/32 | 1/6 | 6 | 0 | 0 | 0 |
| slow_yield | lag_mild | phase05_1802 | 0.2658 | [0.000000, 0.083659] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| slow_yield | lag_mild | phase10_1802 | 0.2781 | [0.000326, 0.086426] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| slow_yield | lag_mild | raw_1812 | -7.5871 | [-1.245451, -0.489250] | 32/32 | 31/32 | 32 | 0 | 0 | 0 |
| slow_yield | lag_mild | nominal_1812 | 0.1967 | [-0.017904, 0.057617] | 19/32 | 6/19 | 19 | 0 | 0 | 0 |
| slow_yield | lag_mild | portfolio0_1812 | 0.6054 | [0.010742, 0.135099] | 7/32 | 2/7 | 7 | 0 | 0 | 0 |
| slow_yield | lag_mild | repair_1812 | 0.1752 | [0.001953, 0.041667] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| slow_yield | lag_mild | phase05_1812 | 0.0077 | [0.000000, 0.002116] | 2/32 | 0/2 | 2 | 0 | 0 | 0 |
| slow_yield | lag_mild | phase10_1812 | 0.0077 | [0.000000, 0.002116] | 2/32 | 0/2 | 2 | 0 | 0 | 0 |
| slow_yield | lag_mild | raw_1822 | -7.8852 | [-1.136084, -0.582658] | 32/32 | 32/32 | 32 | 0 | 0 | 0 |
| slow_yield | lag_mild | nominal_1822 | 0.8789 | [0.030924, 0.176115] | 19/32 | 2/19 | 19 | 0 | 0 | 0 |
| slow_yield | lag_mild | portfolio0_1822 | 0.4978 | [0.001953, 0.122396] | 4/32 | 1/4 | 4 | 0 | 0 | 0 |
| slow_yield | lag_mild | repair_1822 | 0.0645 | [-0.005208, 0.023763] | 5/32 | 1/5 | 5 | 0 | 0 | 0 |
| slow_yield | lag_mild | phase05_1822 | 0.0077 | [0.000000, 0.002116] | 2/32 | 0/2 | 2 | 0 | 0 | 0 |
| slow_yield | lag_mild | phase10_1822 | 0.0077 | [0.000000, 0.002116] | 2/32 | 0/2 | 2 | 0 | 0 | 0 |
| slow_yield | lag_mild | mamps_shared_1802 | -7.6148 | [-1.244173, -0.441073] | 32/32 | 31/32 | 32 | 0 | 0 | 0 |
| slow_yield | lag_mild | mps_shared_1802 | -5.6235 | [-0.832682, -0.364583] | 32/32 | 31/32 | 32 | 0 | 0 | 0 |
| slow_yield | lag_mild | mamps_shared_1812 | -3.8105 | [-0.561548, -0.242179] | 32/32 | 30/32 | 32 | 0 | 0 | 0 |
| slow_yield | lag_mild | mps_shared_1812 | -3.9365 | [-0.591488, -0.247070] | 32/32 | 30/32 | 32 | 0 | 0 | 0 |
| slow_yield | lag_mild | mamps_shared_1822 | -7.0309 | [-0.933268, -0.575187] | 32/32 | 32/32 | 32 | 0 | 0 | 0 |
| slow_yield | lag_mild | mps_shared_1822 | -6.9510 | [-0.945321, -0.550456] | 32/32 | 32/32 | 32 | 0 | 0 | 0 |
| slow_yield | lag_strong | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 3 | 3 | 32 |
| slow_yield | lag_strong | raw_1802 | -11.7936 | [-3.500985, 0.606779] | 32/32 | 27/32 | 32 | 5 | 3 | 32 |
| slow_yield | lag_strong | nominal_1802 | -3.2337 | [-1.278337, 0.045898] | 15/32 | 6/15 | 15 | 4 | 3 | 32 |
| slow_yield | lag_strong | portfolio0_1802 | 0.0208 | [0.000000, 0.007161] | 2/32 | 0/2 | 2 | 3 | 3 | 32 |
| slow_yield | lag_strong | repair_1802 | 0.0000 | [-0.001953, 0.001953] | 2/32 | 1/2 | 2 | 3 | 3 | 32 |
| slow_yield | lag_strong | phase05_1802 | 0.0052 | [0.000000, 0.001953] | 1/32 | 0/1 | 1 | 3 | 3 | 32 |
| slow_yield | lag_strong | phase10_1802 | 0.0052 | [0.000000, 0.001953] | 1/32 | 0/1 | 1 | 3 | 3 | 32 |
| slow_yield | lag_strong | raw_1812 | -8.7029 | [-3.349715, 1.109782] | 32/32 | 26/32 | 32 | 4 | 3 | 32 |
| slow_yield | lag_strong | nominal_1812 | 7.2966 | [0.028304, 2.222013] | 14/32 | 3/14 | 14 | 1 | 3 | 32 |
| slow_yield | lag_strong | portfolio0_1812 | 0.5849 | [0.008138, 0.163086] | 7/32 | 1/7 | 7 | 3 | 3 | 32 |
| slow_yield | lag_strong | repair_1812 | 0.3522 | [0.001628, 0.118001] | 6/32 | 1/6 | 6 | 3 | 3 | 32 |
| slow_yield | lag_strong | phase05_1812 | 0.2443 | [0.000000, 0.091146] | 2/32 | 0/2 | 2 | 3 | 3 | 32 |
| slow_yield | lag_strong | phase10_1812 | 0.2443 | [0.000000, 0.091146] | 2/32 | 0/2 | 2 | 3 | 3 | 32 |
| slow_yield | lag_strong | raw_1822 | -30.0026 | [-6.215649, -1.351839] | 32/32 | 30/32 | 32 | 11 | 3 | 32 |
| slow_yield | lag_strong | nominal_1822 | 7.3512 | [0.008130, 2.255534] | 14/32 | 2/14 | 14 | 1 | 3 | 32 |
| slow_yield | lag_strong | portfolio0_1822 | 4.1383 | [0.030599, 1.404297] | 6/32 | 0/6 | 6 | 2 | 3 | 32 |
| slow_yield | lag_strong | repair_1822 | 0.5147 | [-0.000651, 0.152995] | 4/32 | 1/4 | 4 | 3 | 3 | 32 |
| slow_yield | lag_strong | phase05_1822 | 0.4315 | [0.000000, 0.138021] | 2/32 | 0/2 | 2 | 3 | 3 | 32 |
| slow_yield | lag_strong | phase10_1822 | 0.4315 | [0.000000, 0.138021] | 2/32 | 0/2 | 2 | 3 | 3 | 32 |
| slow_yield | lag_strong | mamps_shared_1802 | -12.0925 | [-3.694784, 0.680062] | 32/32 | 25/32 | 32 | 6 | 3 | 32 |
| slow_yield | lag_strong | mps_shared_1802 | -1.2711 | [-1.586662, 1.464901] | 32/32 | 26/32 | 32 | 2 | 3 | 32 |
| slow_yield | lag_strong | mamps_shared_1812 | -7.5097 | [-3.142000, 1.212573] | 32/32 | 26/32 | 32 | 4 | 3 | 32 |
| slow_yield | lag_strong | mps_shared_1812 | -4.3047 | [-2.206877, 1.102547] | 32/32 | 26/32 | 32 | 3 | 3 | 32 |
| slow_yield | lag_strong | mamps_shared_1822 | -32.2849 | [-6.569661, -1.532699] | 32/32 | 30/32 | 32 | 11 | 3 | 32 |
| slow_yield | lag_strong | mps_shared_1822 | -30.5277 | [-6.293058, -1.355754] | 32/32 | 30/32 | 32 | 11 | 3 | 32 |
| asymmetric_speed | nominal | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| asymmetric_speed | nominal | raw_1802 | -2.0580 | [-0.386719, -0.093750] | 29/32 | 27/29 | 32 | 0 | 0 | 0 |
| asymmetric_speed | nominal | nominal_1802 | 0.5760 | [0.031250, 0.087891] | 15/32 | 1/15 | 15 | 0 | 0 | 0 |
| asymmetric_speed | nominal | portfolio0_1802 | 0.0388 | [0.000000, 0.009115] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| asymmetric_speed | nominal | repair_1802 | 0.0647 | [0.001953, 0.012044] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| asymmetric_speed | nominal | phase05_1802 | 0.0680 | [0.001953, 0.012695] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| asymmetric_speed | nominal | phase10_1802 | 0.0680 | [0.001953, 0.012695] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| asymmetric_speed | nominal | raw_1812 | -0.9060 | [-0.254557, 0.022786] | 28/32 | 16/28 | 32 | 0 | 0 | 0 |
| asymmetric_speed | nominal | nominal_1812 | 0.5048 | [0.025391, 0.079427] | 15/32 | 1/15 | 15 | 0 | 0 | 0 |
| asymmetric_speed | nominal | portfolio0_1812 | 0.0971 | [0.000000, 0.027344] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| asymmetric_speed | nominal | repair_1812 | 0.0485 | [0.000651, 0.010417] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | nominal | phase05_1812 | 0.0518 | [0.000651, 0.010742] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | nominal | phase10_1812 | 0.0518 | [0.000651, 0.010742] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | nominal | raw_1822 | -1.6179 | [-0.277376, -0.071615] | 28/32 | 24/28 | 32 | 0 | 0 | 0 |
| asymmetric_speed | nominal | nominal_1822 | 0.4530 | [0.025391, 0.070964] | 18/32 | 0/18 | 18 | 0 | 0 | 0 |
| asymmetric_speed | nominal | portfolio0_1822 | 0.0906 | [0.000000, 0.026042] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| asymmetric_speed | nominal | repair_1822 | 0.0356 | [0.000968, 0.006836] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | nominal | phase05_1822 | 0.0275 | [0.000326, 0.005859] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| asymmetric_speed | nominal | phase10_1822 | 0.0340 | [0.000810, 0.006510] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | nominal | mamps_shared_1802 | -1.4238 | [-0.291048, -0.045573] | 29/32 | 24/29 | 29 | 0 | 0 | 0 |
| asymmetric_speed | nominal | mps_shared_1802 | -1.3267 | [-0.283203, -0.035807] | 29/32 | 24/29 | 29 | 0 | 0 | 0 |
| asymmetric_speed | nominal | mamps_shared_1812 | -0.7119 | [-0.233740, 0.039714] | 28/32 | 14/28 | 28 | 0 | 0 | 0 |
| asymmetric_speed | nominal | mps_shared_1812 | -3.1096 | [-0.974300, 0.042318] | 28/32 | 14/28 | 28 | 1 | 0 | 0 |
| asymmetric_speed | nominal | mamps_shared_1822 | -0.7572 | [-0.194678, 0.016276] | 28/32 | 16/28 | 28 | 0 | 0 | 0 |
| asymmetric_speed | nominal | mps_shared_1822 | -0.7572 | [-0.194678, 0.016276] | 28/32 | 16/28 | 28 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | raw_1802 | -2.5263 | [-0.455745, -0.121729] | 29/32 | 26/29 | 32 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | nominal_1802 | 0.3295 | [0.000651, 0.069661] | 16/32 | 6/16 | 16 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | portfolio0_1802 | 0.0582 | [0.000000, 0.016276] | 2/32 | 0/2 | 2 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | repair_1802 | 0.1098 | [0.001790, 0.026693] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | phase05_1802 | 0.1050 | [0.001302, 0.026204] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | phase10_1802 | 0.1050 | [0.001302, 0.026204] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | raw_1812 | -0.6009 | [-0.203125, 0.046875] | 28/32 | 16/28 | 32 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | nominal_1812 | 0.3618 | [0.011068, 0.066406] | 16/32 | 7/16 | 16 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | portfolio0_1812 | 0.1357 | [0.001953, 0.029297] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | repair_1812 | 0.1712 | [0.003906, 0.035645] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | phase05_1812 | 0.1018 | [0.000977, 0.026042] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | phase10_1812 | 0.1567 | [0.002279, 0.034342] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | raw_1822 | -1.2922 | [-0.238949, -0.041667] | 28/32 | 22/28 | 32 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | nominal_1822 | 0.4200 | [0.019531, 0.068359] | 18/32 | 3/18 | 18 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | portfolio0_1822 | 0.0582 | [0.000000, 0.016276] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | repair_1822 | 0.1470 | [0.003581, 0.030278] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | phase05_1822 | 0.0840 | [0.000977, 0.020508] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | phase10_1822 | 0.1357 | [0.002279, 0.029297] | 7/32 | 0/7 | 7 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | mamps_shared_1802 | -1.2793 | [-0.277995, -0.029948] | 29/32 | 23/29 | 29 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | mps_shared_1802 | -1.1307 | [-0.263672, -0.017578] | 29/32 | 23/29 | 29 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | mamps_shared_1812 | -0.4135 | [-0.174479, 0.057943] | 28/32 | 14/28 | 28 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | mps_shared_1812 | -0.3554 | [-0.160807, 0.057943] | 28/32 | 14/28 | 28 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | mamps_shared_1822 | -0.5557 | [-0.165381, 0.027995] | 28/32 | 15/28 | 28 | 0 | 0 | 0 |
| asymmetric_speed | gain_drag | mps_shared_1822 | -0.5557 | [-0.165381, 0.027995] | 28/32 | 15/28 | 28 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | raw_1802 | -2.2068 | [-0.430990, -0.095036] | 29/32 | 27/29 | 32 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | nominal_1802 | 0.5954 | [0.033203, 0.089844] | 16/32 | 0/16 | 16 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | portfolio0_1802 | 0.0582 | [0.001302, 0.012370] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | repair_1802 | 0.0744 | [0.001953, 0.014331] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | phase05_1802 | 0.0356 | [0.000000, 0.008138] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | phase10_1802 | 0.0680 | [0.001302, 0.013672] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | raw_1812 | -0.9578 | [-0.260417, 0.018229] | 28/32 | 16/28 | 32 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | nominal_1812 | 0.4336 | [0.016927, 0.072917] | 16/32 | 2/16 | 16 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | portfolio0_1812 | 0.0388 | [0.000651, 0.008464] | 4/32 | 0/4 | 4 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | repair_1812 | 0.0922 | [0.002279, 0.017415] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | phase05_1812 | 0.0582 | [0.000977, 0.011727] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | phase10_1812 | 0.0841 | [0.001953, 0.016276] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | raw_1822 | -1.6050 | [-0.278011, -0.070313] | 28/32 | 24/28 | 32 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | nominal_1822 | 0.5080 | [0.027010, 0.080078] | 20/32 | 2/20 | 20 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | portfolio0_1822 | 0.0129 | [0.000000, 0.003255] | 2/32 | 0/2 | 2 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | repair_1822 | 0.0291 | [0.000651, 0.005859] | 5/32 | 0/5 | 5 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | phase05_1822 | 0.0129 | [0.000000, 0.003581] | 2/32 | 0/2 | 2 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | phase10_1822 | 0.0194 | [0.000000, 0.004557] | 3/32 | 0/3 | 3 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | mamps_shared_1802 | -1.4755 | [-0.322266, -0.037760] | 29/32 | 24/29 | 29 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | mps_shared_1802 | -1.3785 | [-0.311865, -0.029932] | 29/32 | 24/29 | 29 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | mamps_shared_1812 | -0.7701 | [-0.239583, 0.034505] | 28/32 | 15/28 | 28 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | mps_shared_1812 | -3.3232 | [-1.007536, 0.033203] | 28/32 | 15/28 | 28 | 1 | 0 | 0 |
| asymmetric_speed | lag_mild | mamps_shared_1822 | -0.7281 | [-0.193042, 0.019539] | 28/32 | 16/28 | 28 | 0 | 0 | 0 |
| asymmetric_speed | lag_mild | mps_shared_1822 | -0.7281 | [-0.193042, 0.019539] | 28/32 | 16/28 | 28 | 0 | 0 | 0 |
| asymmetric_speed | lag_strong | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | raw_1802 | -3.0058 | [-1.226888, 0.197917] | 28/32 | 14/28 | 32 | 1 | 0 | 32 |
| asymmetric_speed | lag_strong | nominal_1802 | 0.7428 | [0.023438, 0.135417] | 18/32 | 8/18 | 18 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | portfolio0_1802 | 0.0252 | [0.000000, 0.005859] | 4/32 | 0/4 | 4 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | repair_1802 | 0.2943 | [0.009115, 0.056315] | 10/32 | 0/10 | 10 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | phase05_1802 | 0.1700 | [0.004541, 0.035807] | 10/32 | 0/10 | 10 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | phase10_1802 | 0.2313 | [0.005859, 0.047038] | 10/32 | 0/10 | 10 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | raw_1812 | -3.4150 | [-1.334961, 0.194661] | 28/32 | 10/28 | 32 | 1 | 0 | 32 |
| asymmetric_speed | lag_strong | nominal_1812 | 0.8876 | [0.052734, 0.136719] | 18/32 | 3/18 | 18 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | portfolio0_1812 | 0.0126 | [0.000000, 0.003255] | 3/32 | 0/3 | 3 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | repair_1812 | 0.3037 | [0.010742, 0.057454] | 11/32 | 0/11 | 11 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | phase05_1812 | 0.0991 | [0.002116, 0.024414] | 8/32 | 0/8 | 8 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | phase10_1812 | 0.2376 | [0.007161, 0.046712] | 11/32 | 0/11 | 11 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | raw_1822 | 0.7869 | [-0.097673, 0.235026] | 28/32 | 9/28 | 32 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | nominal_1822 | 1.1331 | [0.053385, 0.198568] | 19/32 | 2/19 | 19 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | portfolio0_1822 | 0.1322 | [0.000000, 0.039063] | 4/32 | 0/4 | 4 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | repair_1822 | 0.3462 | [0.012366, 0.063155] | 11/32 | 0/11 | 11 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | phase05_1822 | 0.0897 | [0.001790, 0.022461] | 8/32 | 0/8 | 8 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | phase10_1822 | 0.2203 | [0.004720, 0.045247] | 10/32 | 0/10 | 10 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | mamps_shared_1802 | -2.9365 | [-1.215837, 0.195312] | 28/32 | 12/28 | 28 | 1 | 0 | 32 |
| asymmetric_speed | lag_strong | mps_shared_1802 | -2.8736 | [-1.206722, 0.200521] | 28/32 | 12/28 | 28 | 1 | 0 | 32 |
| asymmetric_speed | lag_strong | mamps_shared_1812 | -2.8295 | [-1.200863, 0.212891] | 28/32 | 9/28 | 28 | 1 | 0 | 32 |
| asymmetric_speed | lag_strong | mps_shared_1812 | -2.8169 | [-1.196956, 0.212891] | 28/32 | 9/28 | 28 | 1 | 0 | 32 |
| asymmetric_speed | lag_strong | mamps_shared_1822 | 0.8246 | [-0.091813, 0.233089] | 28/32 | 9/28 | 28 | 0 | 0 | 32 |
| asymmetric_speed | lag_strong | mps_shared_1822 | 0.7491 | [-0.115251, 0.233073] | 28/32 | 9/28 | 28 | 0 | 0 | 32 |
| tight_headway | nominal | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| tight_headway | nominal | raw_1802 | -4.6743 | [-1.224349, 0.141943] | 32/32 | 18/32 | 32 | 1 | 0 | 0 |
| tight_headway | nominal | nominal_1802 | 1.7184 | [0.081380, 0.249349] | 22/32 | 0/22 | 22 | 0 | 0 | 0 |
| tight_headway | nominal | portfolio0_1802 | 0.8644 | [0.020166, 0.156250] | 10/32 | 1/10 | 10 | 0 | 0 | 0 |
| tight_headway | nominal | repair_1802 | 1.0997 | [0.034505, 0.180827] | 13/32 | 0/13 | 13 | 0 | 0 | 0 |
| tight_headway | nominal | phase05_1802 | 0.8488 | [0.020182, 0.149748] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | nominal | phase10_1802 | 0.9115 | [0.024577, 0.154639] | 10/32 | 0/10 | 10 | 0 | 0 | 0 |
| tight_headway | nominal | raw_1812 | 1.3141 | [0.040039, 0.215177] | 32/32 | 16/32 | 32 | 0 | 0 | 0 |
| tight_headway | nominal | nominal_1812 | 1.4291 | [0.065747, 0.208984] | 23/32 | 0/23 | 23 | 0 | 0 | 0 |
| tight_headway | nominal | portfolio0_1812 | 0.8644 | [0.021484, 0.154297] | 11/32 | 0/11 | 11 | 0 | 0 | 0 |
| tight_headway | nominal | repair_1812 | 1.1015 | [0.032711, 0.180827] | 13/32 | 0/13 | 13 | 0 | 0 | 0 |
| tight_headway | nominal | phase05_1812 | 0.9551 | [0.023596, 0.164714] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| tight_headway | nominal | phase10_1812 | 0.9551 | [0.023596, 0.164714] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| tight_headway | nominal | raw_1822 | 0.6518 | [-0.072266, 0.190763] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| tight_headway | nominal | nominal_1822 | 1.7463 | [0.086914, 0.250326] | 26/32 | 0/26 | 26 | 0 | 0 | 0 |
| tight_headway | nominal | portfolio0_1822 | 1.1294 | [0.033854, 0.188151] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | nominal | repair_1822 | 1.2601 | [0.046224, 0.199386] | 12/32 | 0/12 | 12 | 0 | 0 | 0 |
| tight_headway | nominal | phase05_1822 | 1.1276 | [0.033687, 0.187341] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | nominal | phase10_1822 | 1.1973 | [0.040035, 0.193852] | 10/32 | 0/10 | 10 | 0 | 0 | 0 |
| tight_headway | nominal | mamps_shared_1802 | 0.4287 | [-0.164063, 0.202490] | 32/32 | 17/32 | 32 | 0 | 0 | 0 |
| tight_headway | nominal | mps_shared_1802 | 0.7424 | [-0.062500, 0.201172] | 32/32 | 18/32 | 32 | 0 | 0 | 0 |
| tight_headway | nominal | mamps_shared_1812 | 1.5197 | [0.059570, 0.231771] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| tight_headway | nominal | mps_shared_1812 | 1.4012 | [0.047852, 0.221029] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| tight_headway | nominal | mamps_shared_1822 | 1.3001 | [-0.000334, 0.240243] | 32/32 | 13/32 | 32 | 0 | 0 | 0 |
| tight_headway | nominal | mps_shared_1822 | 1.4361 | [0.018229, 0.249365] | 32/32 | 12/32 | 32 | 0 | 0 | 0 |
| tight_headway | gain_drag | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| tight_headway | gain_drag | raw_1802 | -3.0533 | [-0.793001, 0.082699] | 32/32 | 21/32 | 32 | 0 | 0 | 0 |
| tight_headway | gain_drag | nominal_1802 | 1.1198 | [0.036458, 0.186198] | 25/32 | 4/25 | 25 | 0 | 0 | 0 |
| tight_headway | gain_drag | portfolio0_1802 | 0.8346 | [0.015625, 0.155615] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | gain_drag | repair_1802 | 0.8503 | [0.020015, 0.156413] | 14/32 | 0/14 | 14 | 0 | 0 | 0 |
| tight_headway | gain_drag | phase05_1802 | 0.3912 | [0.004069, 0.088875] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | gain_drag | phase10_1802 | 0.5842 | [0.012695, 0.110352] | 11/32 | 0/11 | 11 | 0 | 0 | 0 |
| tight_headway | gain_drag | raw_1812 | 0.9807 | [0.003255, 0.189453] | 32/32 | 14/32 | 32 | 0 | 0 | 0 |
| tight_headway | gain_drag | nominal_1812 | 1.2867 | [0.048828, 0.203125] | 25/32 | 3/25 | 25 | 0 | 0 | 0 |
| tight_headway | gain_drag | portfolio0_1812 | 0.7859 | [0.011719, 0.152344] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | gain_drag | repair_1812 | 0.8642 | [0.019202, 0.159351] | 12/32 | 0/12 | 12 | 0 | 0 | 0 |
| tight_headway | gain_drag | phase05_1812 | 0.5251 | [0.007812, 0.104980] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | gain_drag | phase10_1812 | 0.5251 | [0.007812, 0.104980] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | gain_drag | raw_1822 | 0.5495 | [-0.078125, 0.170573] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| tight_headway | gain_drag | nominal_1822 | 1.0398 | [0.021810, 0.183936] | 24/32 | 5/24 | 24 | 0 | 0 | 0 |
| tight_headway | gain_drag | portfolio0_1822 | 1.0780 | [0.026042, 0.187500] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| tight_headway | gain_drag | repair_1822 | 1.1024 | [0.029293, 0.189290] | 13/32 | 0/13 | 13 | 0 | 0 | 0 |
| tight_headway | gain_drag | phase05_1822 | 0.7685 | [0.014648, 0.143066] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| tight_headway | gain_drag | phase10_1822 | 0.7755 | [0.015462, 0.143555] | 10/32 | 0/10 | 10 | 0 | 0 | 0 |
| tight_headway | gain_drag | mamps_shared_1802 | 0.1739 | [-0.177751, 0.167334] | 32/32 | 18/32 | 32 | 0 | 0 | 0 |
| tight_headway | gain_drag | mps_shared_1802 | 0.5425 | [-0.077474, 0.173828] | 32/32 | 18/32 | 32 | 0 | 0 | 0 |
| tight_headway | gain_drag | mamps_shared_1812 | 1.0989 | [0.014974, 0.203792] | 32/32 | 13/32 | 32 | 0 | 0 | 0 |
| tight_headway | gain_drag | mps_shared_1812 | 1.0919 | [0.010417, 0.207682] | 32/32 | 12/32 | 32 | 0 | 0 | 0 |
| tight_headway | gain_drag | mamps_shared_1822 | 0.9389 | [-0.028011, 0.197917] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| tight_headway | gain_drag | mps_shared_1822 | 1.0398 | [-0.014006, 0.205094] | 32/32 | 14/32 | 32 | 0 | 0 | 0 |
| tight_headway | lag_mild | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 0 | 0 | 0 |
| tight_headway | lag_mild | raw_1802 | -4.9350 | [-1.247550, 0.114591] | 32/32 | 18/32 | 32 | 1 | 0 | 0 |
| tight_headway | lag_mild | nominal_1802 | 1.5694 | [0.065747, 0.240894] | 23/32 | 3/23 | 23 | 0 | 0 | 0 |
| tight_headway | lag_mild | portfolio0_1802 | 0.7324 | [0.016276, 0.134766] | 10/32 | 1/10 | 10 | 0 | 0 | 0 |
| tight_headway | lag_mild | repair_1802 | 0.8667 | [0.010579, 0.162113] | 11/32 | 0/11 | 11 | 0 | 0 | 0 |
| tight_headway | lag_mild | phase05_1802 | 0.7882 | [0.002767, 0.153813] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | lag_mild | phase10_1802 | 0.7882 | [0.002767, 0.153813] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | lag_mild | raw_1812 | 1.2241 | [0.029614, 0.209961] | 32/32 | 17/32 | 32 | 0 | 0 | 0 |
| tight_headway | lag_mild | nominal_1812 | 1.3288 | [0.050456, 0.211589] | 21/32 | 1/21 | 21 | 0 | 0 | 0 |
| tight_headway | lag_mild | portfolio0_1812 | 0.7603 | [0.011068, 0.143229] | 10/32 | 0/10 | 10 | 0 | 0 | 0 |
| tight_headway | lag_mild | repair_1812 | 0.8492 | [0.007812, 0.162122] | 9/32 | 0/9 | 9 | 0 | 0 | 0 |
| tight_headway | lag_mild | phase05_1812 | 0.7795 | [0.001953, 0.152995] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| tight_headway | lag_mild | phase10_1812 | 0.7865 | [0.001953, 0.153650] | 6/32 | 0/6 | 6 | 0 | 0 | 0 |
| tight_headway | lag_mild | raw_1822 | 0.5197 | [-0.086271, 0.182308] | 32/32 | 16/32 | 32 | 0 | 0 | 0 |
| tight_headway | lag_mild | nominal_1822 | 1.6636 | [0.079427, 0.244149] | 26/32 | 1/26 | 26 | 0 | 0 | 0 |
| tight_headway | lag_mild | portfolio0_1822 | 1.1509 | [0.029948, 0.198568] | 10/32 | 0/10 | 10 | 0 | 0 | 0 |
| tight_headway | lag_mild | repair_1822 | 1.0306 | [0.024740, 0.183443] | 11/32 | 0/11 | 11 | 0 | 0 | 0 |
| tight_headway | lag_mild | phase05_1822 | 0.9451 | [0.017578, 0.175134] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | lag_mild | phase10_1822 | 0.9521 | [0.017578, 0.176758] | 8/32 | 0/8 | 8 | 0 | 0 | 0 |
| tight_headway | lag_mild | mamps_shared_1802 | -1.2799 | [-0.620142, 0.186523] | 32/32 | 16/32 | 32 | 0 | 0 | 0 |
| tight_headway | lag_mild | mps_shared_1802 | 0.5685 | [-0.079435, 0.181641] | 32/32 | 17/32 | 32 | 0 | 0 | 0 |
| tight_headway | lag_mild | mamps_shared_1812 | 1.1404 | [0.010091, 0.209644] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| tight_headway | lag_mild | mps_shared_1812 | 1.1056 | [0.009115, 0.205420] | 32/32 | 15/32 | 32 | 0 | 0 | 0 |
| tight_headway | lag_mild | mamps_shared_1822 | 1.1474 | [-0.014648, 0.229175] | 32/32 | 12/32 | 32 | 0 | 0 | 0 |
| tight_headway | lag_mild | mps_shared_1822 | 1.2799 | [0.001628, 0.236979] | 32/32 | 13/32 | 32 | 0 | 0 | 0 |
| tight_headway | lag_strong | baseline | 0.0000 | [0.000000, 0.000000] | 0/32 | 0/0 | 0 | 2 | 2 | 32 |
| tight_headway | lag_strong | raw_1802 | 10.2178 | [0.020817, 2.600928] | 32/32 | 20/32 | 32 | 0 | 2 | 32 |
| tight_headway | lag_strong | nominal_1802 | 10.6150 | [0.134749, 2.596354] | 24/32 | 6/24 | 24 | 0 | 2 | 32 |
| tight_headway | lag_strong | portfolio0_1802 | 10.0433 | [0.078776, 2.537109] | 15/32 | 2/15 | 15 | 0 | 2 | 32 |
| tight_headway | lag_strong | repair_1802 | 9.9877 | [0.083171, 2.529639] | 16/32 | 1/16 | 16 | 0 | 2 | 32 |
| tight_headway | lag_strong | phase05_1802 | 9.6943 | [0.048828, 2.500016] | 12/32 | 1/12 | 12 | 0 | 2 | 32 |
| tight_headway | lag_strong | phase10_1802 | 9.6943 | [0.048828, 2.500016] | 12/32 | 1/12 | 12 | 0 | 2 | 32 |
| tight_headway | lag_strong | raw_1812 | 11.7343 | [0.253239, 2.729183] | 32/32 | 14/32 | 32 | 0 | 2 | 32 |
| tight_headway | lag_strong | nominal_1812 | 10.3683 | [0.113916, 2.570361] | 25/32 | 10/25 | 25 | 0 | 2 | 32 |
| tight_headway | lag_strong | portfolio0_1812 | 5.3195 | [0.030583, 1.587240] | 11/32 | 1/11 | 11 | 1 | 2 | 32 |
| tight_headway | lag_strong | repair_1812 | 5.3827 | [0.036617, 1.593429] | 17/32 | 3/17 | 17 | 1 | 2 | 32 |
| tight_headway | lag_strong | phase05_1812 | 5.2353 | [0.023438, 1.579443] | 9/32 | 0/9 | 9 | 1 | 2 | 32 |
| tight_headway | lag_strong | phase10_1812 | 5.1841 | [0.017415, 1.573910] | 11/32 | 2/11 | 11 | 1 | 2 | 32 |
| tight_headway | lag_strong | raw_1822 | 6.5230 | [-0.850277, 2.271501] | 32/32 | 16/32 | 32 | 1 | 2 | 32 |
| tight_headway | lag_strong | nominal_1822 | 10.7233 | [0.148437, 2.620475] | 29/32 | 8/29 | 29 | 0 | 2 | 32 |
| tight_headway | lag_strong | portfolio0_1822 | 5.4700 | [0.033854, 1.610693] | 13/32 | 3/13 | 13 | 1 | 2 | 32 |
| tight_headway | lag_strong | repair_1822 | 5.6550 | [0.054199, 1.627934] | 17/32 | 2/17 | 17 | 1 | 2 | 32 |
| tight_headway | lag_strong | phase05_1822 | 5.5211 | [0.042969, 1.615251] | 11/32 | 1/11 | 11 | 1 | 2 | 32 |
| tight_headway | lag_strong | phase10_1822 | 5.5166 | [0.042969, 1.615251] | 12/32 | 2/12 | 12 | 1 | 2 | 32 |
| tight_headway | lag_strong | mamps_shared_1802 | 10.3713 | [0.040299, 2.617855] | 32/32 | 14/32 | 32 | 0 | 2 | 32 |
| tight_headway | lag_strong | mps_shared_1802 | 11.0031 | [0.161125, 2.661483] | 32/32 | 14/32 | 32 | 0 | 2 | 32 |
| tight_headway | lag_strong | mamps_shared_1812 | 11.4093 | [0.210929, 2.698918] | 32/32 | 13/32 | 32 | 0 | 2 | 32 |
| tight_headway | lag_strong | mps_shared_1812 | 11.3431 | [0.213542, 2.689469] | 32/32 | 13/32 | 32 | 0 | 2 | 32 |
| tight_headway | lag_strong | mamps_shared_1822 | 7.5159 | [-0.712573, 2.352865] | 32/32 | 11/32 | 32 | 1 | 2 | 32 |
| tight_headway | lag_strong | mps_shared_1822 | 7.1308 | [-0.755884, 2.317749] | 32/32 | 10/32 | 32 | 1 | 2 | 32 |

## Added physical-gate callback timing

Includes terminal error calls. Concurrent desktop load; old controller timings are not paired speed comparisons. These callback observations do not establish hard-real-time or physical actuation-delay safety.

| Study | Method | Executions | Calls | Error calls | P95 ms | Maximum ms | Over 500 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| phase | mamps_shared_1802 | 512 | 11559 | 0 | 31.081 | 67.710 | 0 |
| phase | mamps_shared_1812 | 512 | 11523 | 0 | 31.672 | 87.785 | 0 |
| phase | mamps_shared_1822 | 512 | 11502 | 0 | 31.441 | 66.113 | 0 |
| phase | mps_shared_1802 | 512 | 11554 | 0 | 30.785 | 73.395 | 0 |
| phase | mps_shared_1812 | 512 | 11523 | 0 | 31.643 | 97.380 | 0 |
| phase | mps_shared_1822 | 512 | 11505 | 0 | 31.569 | 74.253 | 0 |
| transfer | mamps_shared_1802 | 640 | 14909 | 8 | 27.205 | 70.399 | 0 |
| transfer | mamps_shared_1812 | 640 | 14721 | 6 | 28.398 | 62.809 | 0 |
| transfer | mamps_shared_1822 | 640 | 14711 | 14 | 28.346 | 442.612 | 0 |
| transfer | mps_shared_1802 | 640 | 14937 | 4 | 26.972 | 62.990 | 0 |
| transfer | mps_shared_1812 | 640 | 14751 | 5 | 28.490 | 247.839 | 0 |
| transfer | mps_shared_1822 | 640 | 14697 | 14 | 28.371 | 220.931 | 0 |
