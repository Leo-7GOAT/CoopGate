"""Stage complete verified data in one draft; publish after manuscript QA.

Resumption uses the recorded exact release ID and immutable asset digests.
No remote asset is overwritten. No scientific result is recomputed here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101'
OUT=ROOT/'runs/top_two_complete_release_20261104'
ARCHIVE=BASE/'completed_top_two_attempt6_archive_20261104'
REPO='Leo-7GOAT/CoopGate'
TAG='mappo-attempt6-complete-20261104'
SOURCE='147d2f0af28508cc30b7745e6ae70ef45867c9f9'
sys.path.insert(0,str(ROOT/'tools'))
from mappo_completed_top_two_attempt6 import complete_evidence


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,v):
    with p.open('x',encoding='utf-8') as f:
        json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def gh(*args):
    result=subprocess.run(['gh',*args],cwd=ROOT,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(dict(args=args,actual_native_exit_code=result.returncode,stderr=result.stderr))
    return json.loads(result.stdout) if result.stdout.strip() else None


def run(paper=None):
    complete_evidence()
    inventory=json.loads((ARCHIVE/'inventory.json').read_bytes())
    native=json.loads((ARCHIVE/'owner_native_exit.json').read_bytes())
    if native['actual_native_exit_code']!=0 or native['inventory_sha256']!=sha(ARCHIVE/'inventory.json'):
        raise ValueError('Native successful archive required')
    if not inventory['all_archived_bytes_verified'] or inventory['full_effective_schedule']!=59392:
        raise ValueError('Whole original scope required')
    OUT.mkdir(exist_ok=True)
    if not (OUT/'created_release.json').exists():
        if paper is not None:raise ValueError('Prepare exact draft first')
        request=dict(tag_name=TAG,target_commitish=SOURCE,draft=True,
            name='Complete genuine MAPPO confidence gate: fixed three-model joint decision passes',
            body='''Complete independent attempt6, with all original criteria and fixed models retained.

6144 stratified calibration initials and 1024 IID mixture confirmation initials; all59392 scheduled method-initial records and original independent controller/physics replays completed. Recovery retained642 checksum-valid original records and actually ran58750 others. Original interrupted root exit and lost in-flight attempts remain UNKNOWN; total physical attempts including lost work are not known. Original alpha0.05/42 remains spent, with no new allocation.

Three genuinely jointly trained MAPPO actor sets, training-only centralized critic. Mean paired cost gains1.115133/1.093357/1.098964%; gain lower bounds0.832641/0.813703/0.814132%; adoption lower bounds24.513347/24.698771/23.957663%; mixture harm upper bounds3.008315/2.927610/3.360588%. All original joint requirements pass for the full fixed family. Harm protects each car's cost/arrival relative to baseline, NOT collision probability. Calibration harm5/5/8 remains; confirmation targets0/0/0 observed harm does not imply zero risk.

Nominal and independently thinned controls share the expanded candidate pool. Thinned harm30/38/32 persists, with actual adoption300/345/306 versus gate294/296/288. Matching is in expectation, not equivalence; descriptive comparisons are not a new certificate. MPS/MAMPS are disclosed shared-oracle raw-proposal adapters, not complete paper reproductions or candidate-pool-matched SOTA comparisons.

Full actual callback P95 is163-186ms and maxima1.87-2.62s, exceeding the500ms period. No communication latency, hard-real-time, raw-policy superiority or centralized-critic causal advantage claim. Fixed-route three-car and declared actuator/noise-distribution scope only. The finite prediction score is not a calibrated statewise safety probability. Earlier failures and spent error remain public.

Extract ALL36 standalone completed_part ZIPs at the repository root, without concatenation. Inventory binds every one of2970 files /985357252 original bytes; all archive members were read back and SHA256 verified. Frozen scientific sources and original plan are included. Earlier training weights and negative studies remain in their existing public releases. The Chinese manuscript reports both evidence and limits; passing this internal joint decision does not establish top-venue readiness.
''')
        write(OUT/'create_request.json',request)
        created=gh('api','--method','POST',f'repos/{REPO}/releases','--input',str(OUT/'create_request.json'))
        write(OUT/'created_release.json',created)
    created=json.loads((OUT/'created_release.json').read_bytes());rid=created['id']
    remote=gh('api',f'repos/{REPO}/releases/{rid}')
    if remote['tag_name']!=TAG:raise ValueError('Wrong recorded release')
    files={item['name']:ARCHIVE/item['name'] for item in inventory['parts']}
    for name in ['inventory.json','owner_native_exit.json']:files[name]=ARCHIVE/name
    for name,path in [('formal_certificate.json',BASE/'formal_mappo_attempt6_top_two_recovery_20261104/analysis/certificate.json'),
                      ('formal_comparisons.json',BASE/'formal_mappo_attempt6_top_two_recovery_20261104/analysis/comparisons.json'),
                      ('complete_results.md',ROOT/'docs/mappo_top_two_attempt6_completed_20261104.md')]:files[name]=path
    for item in inventory['parts']:
        if sha(files[item['name']])!=item['sha256']:raise ValueError('Archive part changed')
    if paper is not None:
        paper=Path(paper).resolve()
        if not paper.is_relative_to(BASE.resolve()):raise ValueError('Project-local verified paper required')
        qa=json.loads((paper/'visual_and_package_qa.json').read_bytes())
        doc=next(paper.glob('*.docx'))
        if not qa['all_pages_visually_reviewed'] or qa['document_sha256']!=sha(doc):
            raise ValueError('Complete visual/native equation/package QA required')
        files['CoopGate_MAPPO_complete_attempt6_20261104.docx']=doc
        files['manuscript_visual_and_package_qa.json']=paper/'visual_and_package_qa.json'
        files['manuscript_build.json']=paper/'build.json'
        files['manuscript_source_implementation_manifest.json']=paper/'source_implementation_manifest.json'
        for name in ['evidence.json','evidence.native.json']:
            files['paper_'+name]=BASE/'paper_top_two_attempt6_evidence_20261104'/name
    assets={v['name']:v for v in remote['assets']}
    for name,path in files.items():
        if name in assets:
            if assets[name]['digest']!='sha256:'+sha(path) or assets[name]['size']!=path.stat().st_size:
                raise ValueError('Existing remote bytes differ; never overwrite '+name)
            continue
        if not remote['draft']:raise ValueError('Do not mutate already published release')
        asset=gh('api','--method','POST',f'https://uploads.github.com/repos/{REPO}/releases/{rid}/assets?name='+quote(name),
                 '-H','Content-Type: application/octet-stream','--input',str(path))
        if asset['digest']!='sha256:'+sha(path) or asset['size']!=path.stat().st_size:
            raise ValueError('Uploaded remote digest differs')
        write(OUT/(name+'.upload.json'),asset)
        print(json.dumps(dict(asset=name,actual_upload_exit_code=0,release_id=rid)),flush=True)
    remote=gh('api',f'repos/{REPO}/releases/{rid}');assets={v['name']:v for v in remote['assets']}
    for name,path in files.items():
        if assets[name]['digest']!='sha256:'+sha(path):raise ValueError('Remote complete manifest differs')
    if paper is None:
        write(OUT/'raw_upload_complete.json',dict(release_id=rid,draft=remote['draft'],all_raw_digests_verified=True,assets=len(files)))
        print(json.dumps(dict(raw_staged=True,release_id=rid,assets=len(files))),flush=True);return
    gh('api','--method','PATCH',f'repos/{REPO}/releases/{rid}','-F','draft=false')
    remote=gh('api',f'repos/{REPO}/releases/{rid}')
    if remote['draft']:raise ValueError('Publication not observed')
    receipt=dict(published=True,release_id=rid,url=remote['html_url'],source_commit=SOURCE,
        all_asset_digests_verified=True,joint_deployment_certified=True,
        complete_effective_schedule=59392,independent_initials=7168,retained_records=642,
        new_actual_sumo_executions=58750,unknown_original_lost_executions=True,
        raw_policy_or_central_critic_causal_superiority_claim=False,
        assets=[dict(name=name,bytes=path.stat().st_size,sha256=sha(path)) for name,path in files.items()])
    write(OUT/'publication_receipt.json',receipt)
    result=gh('api','--method','POST',f'https://uploads.github.com/repos/{REPO}/releases/{rid}/assets?name=publication_receipt.json',
              '-H','Content-Type: application/octet-stream','--input',str(OUT/'publication_receipt.json'))
    if result['digest']!='sha256:'+sha(OUT/'publication_receipt.json'):raise ValueError('Receipt digest differs')
    print(json.dumps({k:v for k,v in receipt.items() if k!='assets'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--paper',type=Path)
    run(parser.parse_args().paper)
