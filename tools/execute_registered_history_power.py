"""Bind the blind planning rule before invoking complete-pilot-only analysis."""
import json
import plan_history_formal_power as p


def run():
    path=p.e.BASE/'history_power_design_registration_20261103.json'
    registration=p.e.read(path)
    if (registration['source_sha256']!=p.e.sha(p.e.ROOT/'tools/plan_history_formal_power.py')
        or registration['protocol_sha256']!=p.e.sha(p.e.ROOT/'docs/history_power_design_20261103_protocol.md')
        or registration['fixed_grid']!=[list(x) for x in p.GRID] or registration['replicates']!=p.B
        or registration['stream']!=p.STREAM or registration['joint_harm_stress']!=p.STRESS
        or registration['pilot_effects_read'] or registration['pilot_report_exists']
        or registration['new_error_allocated']):
        raise ValueError('Blind source/assumption/scope registration changed')
    p.run()
    p.e.write(p.OUT/'execution_registration_binding.json',dict(registration_sha256=p.e.sha(path),
        report_sha256=p.e.sha(p.OUT/'report.json'),wrapper_sha256=p.e.sha(__file__),
        no_new_error_allocated=True,rule_fixed_before_pilot_completion=True))


if __name__=='__main__':run()
