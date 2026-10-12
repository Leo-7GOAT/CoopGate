"""Copy only designated reproducibility outputs into version-controlled artifacts."""
from pathlib import Path
import shutil
import json
import hashlib
import csv

root=Path(__file__).resolve().parents[1]
destination=root/'artifacts'
destination.mkdir(exist_ok=True)
for seed in (7,19):
    train=root/'runs'/f'train50k{seed}'
    evaluation=root/'runs'/f'eval50k{seed}'
    out=destination/f'seed{seed}'
    out.mkdir(exist_ok=True)
    for name in ('policy.pt','training.csv','config.json'):
        shutil.copy2(train/name,out/name)
    for file in evaluation.iterdir():
        if file.suffix in ('.json','.csv','.png','.pdf'):
            shutil.copy2(file,out/file.name)
# Canonical newlines preserve numeric contents and make release hashes portable.
for file in destination.rglob('*'):
    if file.suffix in ('.json', '.csv') and file.name != 'SHA256SUMS.json':
        file.write_bytes(file.read_bytes().replace(b'\r\n', b'\n'))
hashes={str(p.relative_to(destination)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(destination/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n', newline='\n')
print(json.dumps({'artifact_files':len(hashes),'directory':str(destination)},indent=2))
