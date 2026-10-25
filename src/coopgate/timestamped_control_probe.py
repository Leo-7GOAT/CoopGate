"""Matched-public-input observation preprocessing for a separate SUMO probe.

The original controller still assumes a point state and lag_strong dynamics.
Nothing here restores its physical certificate or certifies performance risk.
"""
from copy import deepcopy

import numpy as np

from .aged_observation_envelope import AgeAwareStateObserver, timestamp_vector
from .bounded_observation_v45 import project_velocity
from .recovery_joint_gate_v35 import RecoveryJointGate

VARIANTS = ('projected_raw', 'constant_velocity', 'envelope_midpoint')


class TimestampPort:
    """A public sensor channel containing exactly profile, index and timestamps."""
    def __init__(self):
        self.latest = None

    def publish(self, evidence):
        decision = evidence['decision']
        if type(decision) is not int or decision < 0:
            raise ValueError('Integer current decision required')
        expected = 0 if self.latest is None else self.latest['decision']+1
        if decision != expected:
            raise ValueError('Sensor delivery order changed')
        profile = evidence['profile']
        if profile not in ('clean_reference','bounded_bias_jitter','stale_dropout'):
            raise ValueError('Unknown public sensor profile')
        stamps = timestamp_vector(evidence['source_decision'], decision)
        self.latest = dict(profile=profile, decision=decision, source_decision=stamps.tolist())

    def current(self, decision):
        if self.latest is None or self.latest['decision'] != decision:
            raise ValueError('No source timestamps for this current decision')
        return deepcopy(self.latest)


class PublicSensorBridge:
    """Sensor-side adapter; only the sanitized port reaches the controller."""
    def __init__(self, sensor, port):
        self.sensor, self.port = sensor, port

    def observe(self, history):
        observed, evidence = self.sensor.observe(history)
        self.port.publish(evidence)
        return observed, evidence


class TimestampedGate:
    def __init__(self, model, profile, rule, variant, port):
        if variant not in VARIANTS:
            raise ValueError('Unknown fixed preprocessing variant')
        self.inner = RecoveryJointGate(model, profile, rule)
        self.c = self.inner.c
        self.variant, self.port = variant, port
        self.observer = None
        self.receipts = []

    def select(self, observed, previous, unused=None):
        decision = len(self.receipts)
        frame = self.port.current(decision)
        raw = np.asarray(observed, dtype=float)
        receipt = dict(variant=self.variant, **frame, raw_state=raw.tolist(),
            previous_command=np.asarray(previous,dtype=float).tolist(), envelope=None,
            processed_state=None, estimator_error=None, inner_error=None,
            restores_physical_certificate=False, performance_risk_certified=False)
        self.receipts.append(receipt)
        try:
            if self.observer is None:
                self.observer = AgeAwareStateObserver(frame['profile'])
            if self.observer.profile != frame['profile']:
                raise ValueError('Sensor profile changed inside one episode')
            envelope = self.observer.update(raw, frame['source_decision'], previous)
            receipt['envelope'] = envelope
            processed = project_velocity(raw)
            if self.variant == 'constant_velocity':
                age = np.asarray(envelope['age_decisions'])*self.c.decision_dt
                processed[:,0] += age*processed[:,1]
            elif self.variant == 'envelope_midpoint':
                processed = (np.asarray(envelope['lower'])+np.asarray(envelope['upper']))/2
            receipt['processed_state'] = processed.tolist()
        except Exception as error:
            receipt['estimator_error'] = dict(type=type(error).__name__,message=str(error))
            raise
        try:
            command, info = self.inner.select(processed, previous, unused)
        except Exception as error:
            receipt['inner_error'] = dict(type=type(error).__name__,message=str(error))
            raise
        return command, {**info, 'age_adapter':deepcopy(receipt)}
