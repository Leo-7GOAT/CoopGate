"""Process-isolated proposals with a prechecked baseline and sticky failover.

This is an experimental software deadline mechanism, not a hard-real-time or
actuation-delay safety theorem. SUMO still advances after select returns. A
deadline changes the deployed policy and does not inherit fault-free calibrated
performance risk. No live prospective controller uses this module.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import hashlib
import multiprocessing as mp
import pickle
import time

import numpy as np

from .fast_v8 import parameters
from .merge_v4 import MergeConfig
from .robust_v9 import robust_certificate, validate

MAX_REPLY_BYTES = 2 * 1024 * 1024


def _send_reply(connection, reply):
    try:
        payload = pickle.dumps(reply, protocol=5)
        if len(payload) > MAX_REPLY_BYTES:
            raise ValueError("Primary reply exceeds transport limit")
    except Exception as exc:
        reply = {k: reply[k] for k in ("token", "step", "identity") if k in reply}
        reply.update(kind="error", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
        payload = pickle.dumps(reply, protocol=5)
    connection.send_bytes(payload)


def _receive_reply(connection):
    # Only our trusted local proposal worker writes this connection.
    return pickle.loads(connection.recv_bytes(maxlength=MAX_REPLY_BYTES))


def fingerprint(step, state, previous):
    digest = hashlib.sha256(str(int(step)).encode("ascii"))
    for value in (state, previous):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _worker(connection, factory):
    try:
        controller = factory()
        _send_reply(connection, {"kind": "ready"})
        while True:
            request = connection.recv()
            if request is None:
                break
            token, step, state, previous = request
            identity = fingerprint(step, state, previous)
            try:
                controller.step = step
                command, info = controller.select(state.copy(), previous.copy())
                reply = dict(kind="ok", token=token, step=step, identity=identity,
                             command=command, info=info)
            except Exception as exc:
                reply = dict(kind="error", token=token, step=step, identity=identity,
                             error=f"{type(exc).__name__}: {str(exc)[:1000]}")
            _send_reply(connection, reply)
    except (EOFError, BrokenPipeError, OSError):
        pass
    except Exception as exc:
        try:
            _send_reply(connection, dict(kind="startup_error", error=f"{type(exc).__name__}: {str(exc)[:1000]}"))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class DeadlineProfile:
    """One worker per episode; cleanup is explicitly outside the control callback.

    Warm-up belongs in factory / baseline preparation. Call prepare before the
    episode, then select, finally close. The baseline is re-evaluated from the
    current measured state and previous executed command before waiting for a
    proposal. A timeout permanently disables primary use in this episode.
    """
    def __init__(self, primary_factory, baseline, admitted=True, budget_seconds=.5,
                 config=None):
        if not np.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError("A finite positive software budget is required")
        if admitted and not callable(primary_factory):
            raise ValueError("Admitted primary requires a spawn-compatible factory")
        self.factory, self.baseline = primary_factory, baseline
        self.accept, self.budget = bool(admitted), float(budget_seconds)
        self.c = config or MergeConfig()
        self.packed = parameters(self.c)
        self.step = 0
        self.fault_fallback = False
        self.fault_reason = ""
        self.primary_used_ever = False
        self.prepared = self.closed = False
        self.process = self.connection = self.executor = self.pending = None
        self.setup_seconds = 0.

    def prepare(self, startup_timeout=30.):
        if self.prepared or self.closed:
            raise RuntimeError("Prepare once before use")
        if not np.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("Invalid startup timeout")
        started = time.monotonic()
        if self.accept:
            context = mp.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(target=_worker, args=(child, self.factory), daemon=True)
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="coopgate-proposal-io")
            self.process.start()
            child.close()
            self.pending = self.executor.submit(_receive_reply, self.connection)
            try:
                ready = self.pending.result(timeout=startup_timeout)
                if ready.get("kind") != "ready":
                    raise RuntimeError("Primary startup failed: " + str(ready))
            except BaseException:
                self.close()
                raise
            self.pending = None
        self.prepared = True
        self.setup_seconds = time.monotonic() - started
        return self

    def _exchange(self, request):
        # Both writing and full-message decoding stay off the control thread.
        self.connection.send(request)
        return _receive_reply(self.connection)

    def _checked(self, state, previous, command, info):
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Nonfinite or malformed command")
        if np.any(command < -self.c.brake-1e-9) or np.any(command > self.c.accel+1e-9):
            raise ValueError("Acceleration limit")
        if np.any(np.abs(command-previous) > self.c.jerk*self.c.decision_dt+1e-9):
            raise ValueError("Command-slew limit")
        if not robust_certificate(state, previous, command, self.packed, .5):
            raise ValueError("Missing physical certificate")
        return command.copy(), dict(info)

    def select(self, state, previous, unused=None):
        if not self.prepared or self.closed:
            raise RuntimeError("Prepared live profile required")
        started = time.monotonic()
        deadline = started + self.budget
        x, prev, _ = validate(state, previous, self.c, .5)
        self.baseline.step = self.step
        # A baseline error propagates; there is no unchecked emergency command.
        baseline = self._checked(x, prev, *self.baseline.select(x.copy(), prev.copy()))
        baseline_ms = 1000 * (time.monotonic() - started)
        chosen, info = baseline
        used, timed_out, waited = False, False, 0.
        if self.accept and not self.fault_fallback:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.fault_reason = "baseline_preparation_exhausted_budget"
                self.fault_fallback = timed_out = True
            else:
                wait_started = time.monotonic()
                identity = fingerprint(self.step, x, prev)
                token = self.step
                self.pending = self.executor.submit(self._exchange, (token, self.step, x.copy(), prev.copy()))
                try:
                    # Recompute after snapshot, hashing and dispatch overhead.
                    reply = self.pending.result(timeout=max(0., deadline-time.monotonic()))
                    if (not isinstance(reply, dict) or reply.get("token") != token or reply.get("step") != self.step
                            or reply.get("identity") != identity):
                        raise ValueError("Stale or mismatched proposal reply")
                    if reply.get("kind") != "ok":
                        raise RuntimeError("Primary failed: " + str(reply.get("error")))
                    candidate = self._checked(x, prev, reply["command"], reply["info"])
                    if time.monotonic() >= deadline:
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
                waited = 1000 * (time.monotonic() - wait_started)
        self.step += 1
        elapsed_ms = 1000 * (time.monotonic() - started)
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

    def close(self):
        if self.closed:
            return getattr(self, "shutdown", {"already_closed": True})
        self.closed = True
        forced = False
        if self.process is not None:
            if self.pending is None and self.process.is_alive():
                try:
                    self.executor.submit(self.connection.send, None).result(timeout=.25)
                except Exception:
                    pass
                self.process.join(timeout=.25)
            if self.process.is_alive():
                forced = True
                self.process.terminate()
                self.process.join(timeout=2.)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2.)
            if self.process.is_alive():
                raise RuntimeError("Worker cleanup did not complete")
        if self.connection is not None:
            self.connection.close()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.shutdown = dict(worker_pid=None if self.process is None else self.process.pid,
                             exit_code=None if self.process is None else self.process.exitcode,
                             forced_termination=forced, worker_alive=False)
        return self.shutdown
