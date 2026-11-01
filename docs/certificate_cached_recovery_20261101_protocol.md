# Exact nominal certificate cache combined with admissible rejection

The preceding full pruning replay completed all160 performance-selector source
episodes,320 whole-controller replays and7,324 calls with identical commands,
adoption and accepted evidence. Paired total callback cost fell26.0% for the
nominal selector and15.9% for the bank selector, but its worst callback still
exceeds1.4s. No new SUMO trial, risk certificate or isolated runtime claim arose.

Prepare an independent static prototype that adds exact caching of the nominal
robust certificate inside forecast planning. Cache keys preserve every bit of
state(6),previous command(3),candidate command(3),radius(1),configuration(12).
Store both true and false pure-function outputs. Reset at every online decision;
capacity65,536 entries. Exhaustion computes the original function normally.
Actuator-memory certificates are never cached or skipped. The actual strong
baseline is unchanged. Record queries,hits,misses,capacity skips and entry count.

Generate separate nominal planner/forecast functions by AST cloning. Only
certificate dispatch and cache arguments change; all arithmetic,candidate
ordering,strict comparisons,costs,physical checks and logical work counts
retain their original syntax. No global monkeypatch or alteration of frozen
controllers is allowed. Commit the generator and its static generated source.
Test exact planner command/value/work equality with zero,tiny and full cache
capacity and separation of every key-input bit family before gate testing.

Smoke/full gate comparisons follow the previous fixed scope exactly:4 source
episodes/8 controller replays first,then all160/320. Both implementations get
their own complete measured callbacks and separate caches,rotating/reversing
order. Compare every command,baseline evidence,adoption,selected composition,
accepted path differences and margin. Preserve failures before aborting. No
extra SUMO trials or independent initials are counted. Runtime may overlap
registered three-model gate collections; no hard-real-time guarantee.

Source scope and original risk/adoption/utility thresholds remain unchanged.
Only after full equivalence and fresh actual SUMO verification could this
prototype replace an experimental deployment implementation. A speedup is not
a performance/risk improvement by itself and exact caching is not claimed as
a new research theorem.
