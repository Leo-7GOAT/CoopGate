"""Prospective power for fixed all-or-none families; no outcome data are read."""
import json
import math
from pathlib import Path
import shutil

from design_risk_power_v20 import exact_cutoffs, joint_lower_bound, single_target_power
from probe_disturbed_gate_v19 import digest, save

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/v20/family_risk_power_design'


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve prospective family-risk design')
    levels = dict(individual_eighteen=.05 / 18, three_fixed_families=.05 / 3, one_fixed_family=.05)
    grid, rows = (512, 1024, 2048, 4096, 8192), []
    for procedure, level in levels.items():
        cutoff = exact_cutoffs(max(grid), .05, level)
        for n in grid:
            for q in (.1, .2, .3):
                for risk in (.005, .01, .02, .03):
                    marginal = single_target_power(n, q, risk, cutoff)
                    rows.append(dict(procedure=procedure, independent_initials=n,
                        hypothetical_adoption_probability=q, hypothetical_conditional_risk=risk,
                        constituent_pass_probability=marginal,
                        all_six_pass_probability_lower_bound=joint_lower_bound([marginal] * 6)))
                if single_target_power(n, q, .05, cutoff) > level + 1e-12:
                    raise ValueError('Constituent boundary-null error exceeded its level')
    OUTPUT.mkdir(parents=True)
    source = ROOT / 'src/coopgate/family_risk_v20.py'
    save(OUTPUT / 'analysis.json', dict(purpose='Prospective alternative claim structures, not re-certification',
        rows=rows, risk_limit=.05, total_error=.05, constituent_levels=levels,
        zero_harm_minimum_adoptions={k:math.ceil(math.log(v) / math.log(.95)) for k,v in levels.items()},
        family_scope='All three fixed models and both conditional risks must pass. Selection of any of F predeclared families uses per-family level 0.05/F.',
        power_scope='Six equal hypothetical marginals; union lower bound on all passing. Does not assume independence or certify gain/adoption.',
        existing_certifications_changed=False, calibration_or_final_outcomes_read=False,
        future_protocol_or_sample_size_selected=False, source_sha256=digest(__file__),
        certificate_source_sha256=digest(source), reference_power_source_sha256=digest(ROOT / 'tools/design_risk_power_v20.py')))
    for path in (Path(__file__), source, ROOT / 'tools/design_risk_power_v20.py'):
        shutil.copy2(path, OUTPUT / path.name)
    save(OUTPUT / 'sha256.json', {p.name:digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(minimum_adoptions={k:math.ceil(math.log(v) / math.log(.95)) for k,v in levels.items()},
        adoption_20percent=[r for r in rows if r['hypothetical_adoption_probability']==.2
                           and r['hypothetical_conditional_risk'] in (.01, .02)
                           and r['independent_initials'] in (1024,2048,4096)])), flush=True)


if __name__ == '__main__':
    run()
