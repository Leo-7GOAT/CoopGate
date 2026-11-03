# Reproducibility repair and retained failed checks

The [initial workflow](https://github.com/Leo-7GOAT/CoopGate/actions/runs/41628223186)
failed during collection because three CSPI analysis tests could not import
pandas. The first repair declared an optional dependency group in pyproject.toml.
That was incorrect: although outside the current study's 163 source hashes,
the manifest is bound by older registered studies. The
[subsequent full run](https://github.com/Leo-7GOAT/CoopGate/actions/runs/41632332900)
retains its result: 35 failed, 1930 passed, 30 skipped and 24 errors. Neither
failed workflow is called successful or removed from the record.

The corrective change restores pyproject.toml byte for byte from commit
3ab1d9fc4375b0371cc91bd22b1c1dccdfa90b3a. Separate
requirements-baselines.txt declares pandas, scikit-learn and threadpoolctl;
pytest.ini declares the repository root, src and tools import paths. CI and
the fresh-environment quick start install the optional requirements separately.
No package was installed or upgraded in either existing local environment.

Two registration unit tests now isolate host library versions in an explicitly
named fixture while retaining real candidate and model identity validation.
A separate negative test requires the production runtime mismatch guard to
reject an incompatible runtime. No production runtime guard or stored
registration is changed. The immutable legacy native-failure test receives
a portable observation provider through conftest.py: it queries only its own
child's published PID. The real child, original acknowledgement checks, actual
wait and exit-code-7 assertions remain. The frozen test itself is unchanged.

The three final joint MAPPO checkpoints are also published directly at their
original paths. Their bytes must match the previously registered actor hashes.
They were already included in the public training archives, but a clean
checkout did not extract those archives automatically, producing 24 fixture
errors. This change neither retrains nor substitutes an actor.

Corrective local verification completed with 92 tests passing in 34.96 seconds
(actual native exit 0, terminal 84da0f), covering historical source guards,
registration checks, actual child exit 7, and MAPPO training tests. An additional
combined command passed 22 CSPI analysis tests in the already existing separate
environment and 17 powered-MAPPO preflights in the frozen simulation environment
(combined actual exit 0, terminal ac2aad). The current study's 163 registered
sources were independently verified unchanged. The original and corrective
receipts remain in artifacts/mappo_ctde_20261101/reproducibility_ci_repair_20261103.

These are targeted checks. Full remote pytest and all historical release
verifiers still require a successful fresh workflow after the correction is
published; their success must not be inferred from these local results.

CI ignores only reserved data-transfer-* intermediate transport branches and
replaces obsolete same-ref CI runs through concurrency. All main/code branch
pushes, pull requests, test assertions and release-verifier steps remain.
Cancelled CI is never counted as a successful scientific run. The separate
locally owned simulation and replay jobs are unaffected. Scientific thresholds,
registered samples and spent error allocations remain unchanged.
