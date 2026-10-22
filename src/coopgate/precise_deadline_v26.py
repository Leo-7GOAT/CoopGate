"""Use one high-resolution clock for the isolated proposal budget and timing.

The completed coarse-clock experiment remains reproducible in DeadlineProfile.
The select body below changes only its clock reads; an AST regression checks
that invariant. Platform scheduling and baseline/validation execution can still
overrun the budget. This is not a hard deadline or a performance-risk certificate.
"""
from concurrent.futures import TimeoutError
import time

from .deadline_profile_v26 import DeadlineProfile, fingerprint
from .robust_v9 import validate


class PreciseDeadlineProfile(DeadlineProfile):
    def prepare(self, startup_timeout=30.):
        started = time.perf_counter()
        super().prepare(startup_timeout)
        self.setup_seconds = time.perf_counter() - started
        return self

    def select(self, state, previous, unused=None):
        if not self.prepared or self.closed:
            raise RuntimeError("Prepared live profile required")
        started = time.perf_counter()
        deadline = started + self.budget
        x, prev, _ = validate(state, previous, self.c, .5)
        self.baseline.step = self.step
        # A baseline error propagates; there is no unchecked emergency command.
        baseline = self._checked(x, prev, *self.baseline.select(x.copy(), prev.copy()))
        baseline_ms = 1000 * (time.perf_counter() - started)
        chosen, info = baseline
        used, timed_out, waited = False, False, 0.
        if self.accept and not self.fault_fallback:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self.fault_reason = "baseline_preparation_exhausted_budget"
                self.fault_fallback = timed_out = True
            else:
                wait_started = time.perf_counter()
                identity = fingerprint(self.step, x, prev)
                token = self.step
                self.pending = self.executor.submit(self._exchange, (token, self.step, x.copy(), prev.copy()))
                try:
                    # Recompute after snapshot, hashing and dispatch overhead.
                    reply = self.pending.result(timeout=max(0., deadline-time.perf_counter()))
                    if (not isinstance(reply, dict) or reply.get("token") != token or reply.get("step") != self.step
                            or reply.get("identity") != identity):
                        raise ValueError("Stale or mismatched proposal reply")
                    if reply.get("kind") != "ok":
                        raise RuntimeError("Primary failed: " + str(reply.get("error")))
                    candidate = self._checked(x, prev, reply["command"], reply["info"])
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("Proposal validation completed after deadline")
                    chosen, info = candidate
                    used = self.primary_used_ever = True
                    self.pending = None
                except TimeoutError:
                    self.fault_reason = "proposal_deadline"
                    self.fault_fallback = timed_out = True
                except (RuntimeError, ValueError, TypeError, IndexError, KeyError, EOFError, OSError) as exc:
                    self.fault_reason = f"{type(exc).__name__}: {exc}"
                    self.fault_fallback = True
                waited = 1000 * (time.perf_counter() - wait_started)
        self.step += 1
        elapsed_ms = 1000 * (time.perf_counter() - started)
        info.update(in_set=True, certified=True, performance_admitted=self.accept,
                    fault_fallback=self.fault_fallback, fault_reason=self.fault_reason,
                    deadline_primary_used=used, deadline_primary_used_ever=self.primary_used_ever,
                    deadline_triggered_this_step=timed_out, deadline_budget_ms=1000*self.budget,
                    deadline_observed_overrun=elapsed_ms > 1000*self.budget,
                    baseline_preparation_ms=baseline_ms, primary_wait_ms=waited,
                    primary_process_pid=None if self.process is None else self.process.pid,
                    gate_ms=elapsed_ms, deadline_setup_ms=1000*self.setup_seconds,
                    deadline_risk_certified=False)
        return chosen, info
