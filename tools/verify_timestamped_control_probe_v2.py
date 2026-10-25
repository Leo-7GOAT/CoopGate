"""Reuse the frozen scientific audit body with the corrected manifest view."""
import execute_timestamped_control_probe_v2 as run
import verify_timestamped_control_probe as original

io=run.io


def configured_verify():
    scopes=run.private(original.controller_scopes,run=run)
    return run.private(original.verify,run=run,controller_scopes=scopes)


def verify():return configured_verify()()
