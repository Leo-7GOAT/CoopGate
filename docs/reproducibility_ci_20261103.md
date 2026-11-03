# Reproducibility checks and missing analysis dependencies

The push workflow at commit3ab1d9fc4375b0371cc91bd22b1c1dccdfa90b3a
[failed during pytest collection](https://github.com/Leo-7GOAT/CoopGate/actions/runs/41628223186).
Checkout and environment setup succeeded. Three existing CSPI analysis tests
could not import pandas;pytest exited2 before any historical release verifier
ran. This is a reproducibility configuration failure,not evidence about the
running MAPPO controller's efficacy.

The analysis code also imports scikit-learn and threadpoolctl. These remain
optional,declared under the new baselines extra in pyproject.toml. CI now
requests dev,fast,baselines in its existing fresh runner installation. Core
runtime dependencies and every existing test/release-verifier step remain.
No dependency was installed or upgraded in either existing local environment.

An isolated local test invocation additionally revealed that one analysis test
relied on another collected test to add tools/ to the import path. The pytest
configuration now declares src/ and tools/ explicitly,so the test entry point
does not depend on collection order. No test assertions were removed or eased.

Using the ALREADY EXISTING separate .venv-baselines environment,the three
previously failing test modules now pass all22tests in4.64s (actual native0,
terminal802bb8). Four warnings originate from unchanged pinned author-code
docstring escapes. The initial isolated collection failure (actual1,90ff86)
is retained in the repair evidence. The frozen simulation environment also
passes all17powered-MAPPO adapter/integrity/failure preflights in18.43s
(actual0,319f63). No local package installation was needed.

The manifest is outside the163registered source hashes,and all163controller/
protocol sources were verified unchanged against the bound original plan.
The source/protocol,actual sampling,scientific thresholds and spent error of
the live45,056-run study remain fixed. The optional-dependency/test-path fix
does not reinterpret any old result or replace a formal experiment.

Full GitHub rerun status must be checked after this fix is public. Local
targeted passes are not a claim that the entire workflow or all historical
release replays have passed. Exact local receipts and original failure links
are underartifacts/mappo_ctde_20261101/reproducibility_ci_repair_20261103.

The quick start now creates a fresh .venv-check and requests all full-suite
extras. Existing experiment environments are not reinstalled. CI also ignores
the publisher's reserved data-transfer-* intermediate blob-transport branches,
while retaining every main/code-branch push and pull-request check. Same-ref
concurrency replaces obsolete checks with the latest complete workflow;all
test assertions and historical release-verifier commands remain unchanged.
Cancelled obsolete CI runs remain recorded and are never called successful
scientific experiments. Main-study data collection and reference replay are
separate locally owned processes and are unaffected by this CI scheduling.
