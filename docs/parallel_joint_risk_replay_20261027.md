# Parallel execution of the existing joint-risk replay

The V42 independent auditor is stateless between initial-condition blocks. The added executor distributes the unchanged audit function over three processes, each with its own fixed CPU model copy and one Torch thread. All 16,384 original episodes remain required. The current OOD replay, every SUMO collector, all controller code and all statistical criteria are unchanged.

Before inference, the original finalizer runs with its original bytecode, original terminal/source/quiet checks and existing tuple-to-list key adapter. Its audit calls must match the complete case, full stored record and model tensor fingerprint of independently recomputed results. Missing, changed, duplicate or unused calls fail before the original inference function can execute. Results retain original order and failures. Original imported module globals are not modified.

The execution contract is registered before V42 binding or collection. A separate committed receipt pins the contract to the original joint-study binding. Workers join before the original quiet-project finalizer executes; interrupted replay prefixes cannot resume implicitly. Per-block audit receipts and full source/input checks remain available for inspection.

The operational equivalence benchmark uses exactly 64 already-completed V36 traces from blocks 106–121, comprising the baseline and three fixed bank policies in each block. Only outer case/method labels are adapted to the V42 record interface; physical traces, calls and recorded rows remain unchanged. Serial and parallel outputs must be exactly equal. This adds zero SUMO runs and zero independent scientific samples. Its timing is a shared-host diagnostic, with process startup included, and does not establish standalone controller latency or WCET.

The fixed benchmark produced exactly equal audit facts and argument fingerprints for all 64 traces: serial 59.0466 seconds, three processes 27.7835 seconds, observed ratio 2.1252. This supports enabling parallel replay for the forthcoming joint study; it does not establish that the full formal replay will achieve that ratio or that any deployment criterion has passed.
