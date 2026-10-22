"""Archive a complete coupled development study, including negative outcomes."""
import argparse,hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def archive(study,analysis,audit,ablation,out,diagnostic=None,forecast=None,gate_diagnostic=None,stress_forecast=None,stress_gate=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for source,name in ((study,'study'),(analysis,'analysis'),(ablation,'actor_ablation')):
        shutil.copytree(source,out/name)
    shutil.copy2(audit,out/'audit.json')
    if diagnostic:shutil.copytree(diagnostic,out/'update_diagnostic')
    if forecast:shutil.copytree(forecast,out/'forecast')
    if gate_diagnostic:shutil.copytree(gate_diagnostic,out/'gate_diagnostic')
    if stress_forecast:shutil.copytree(stress_forecast,out/'stress_forecast')
    if stress_gate:shutil.copytree(stress_gate,out/'stress_gate')
    for name in ('ablate_cooperative_v15.py','analyze_cooperative_v15.py','verify_cooperative_v16.py','diagnose_updates_v16.py'):
        target=out/'analysis_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'tools'/name,target)
    manifest={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(destination=str(out),assets=len(manifest))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('study','analysis','audit','ablation','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--diagnostic');p.add_argument('--forecast');p.add_argument('--gate-diagnostic')
    p.add_argument('--stress-forecast');p.add_argument('--stress-gate')
    a=p.parse_args();archive(a.study,a.analysis,a.audit,a.ablation,a.out,a.diagnostic,a.forecast,a.gate_diagnostic,a.stress_forecast,a.stress_gate)
