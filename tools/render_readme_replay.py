"""Render recorded SUMO longitudinal states, not a new experiment or GUI capture.

Default: rebuild from the small published extract. --extract requires restored
attempt6 data and selects the first non-adopted and first adopted seed4703 case
in shard0 schedule order, without ranking cost or gain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'artifacts/mappo_ctde_20261101/formal_mappo_attempt6_top_two_recovery_20261104'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def unpack(blob,digest):
    raw=zlib.decompress(blob)
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Record checksum differs')
    return json.loads(raw)


def extract(target):
    if target.exists():raise FileExistsError('Keep original extract')
    selected={}
    for db in sorted((STUDY/'confirmation/shard0').glob('block*/observations.sqlite')):
        seal=json.loads(Path(str(db)+'.seal.json').read_bytes())
        if sha(db)!=seal['database_sha256']:raise ValueError('Sealed database differs')
        with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as connection:
            rows=connection.execute('SELECT case_key,row_blob,trace_blob,row_sha256,trace_sha256 FROM episodes ORDER BY position').fetchall()
        for key,rb,tb,rh,th in rows:
            cid,method=json.loads(key)
            if method!='4703':continue
            row=unpack(rb,rh);trace=unpack(tb,th)
            kind='learning_adopted' if any(t['actual_learning_used'] for t in trace) else 'baseline_only'
            if kind in selected:continue
            baseline=next(v for v in rows if json.loads(v[0])==[cid,'baseline'])
            br=unpack(baseline[1],baseline[3]);bt=unpack(baseline[2],baseline[4])
            if row['row']['initial']!=br['row']['initial']:raise ValueError('Unmatched initial states')
            def compact(values):
                return [dict(t=v['t'],x=v['x'],command=v['command'],learning=v['actual_learning_used']) for v in values]
            selected[kind]=dict(case_id=cid,seed=row['row']['seed'],scenario=row['row']['scenario'],
                plant=row['row']['plant_name'],initial=row['row']['initial'],
                target=compact(trace),baseline=compact(bt),database=db.relative_to(ROOT).as_posix(),
                database_sha256=sha(db),target_row_sha256=rh,target_trace_sha256=th,
                baseline_row_sha256=baseline[3],baseline_trace_sha256=baseline[4])
        if len(selected)==2:break
    if len(selected)!=2:raise ValueError('Both demonstration types required')
    target.write_text(json.dumps(dict(selection='First of each adoption type in shard0 schedule order, fixed actor4703; no gain ranking',
        source='Completed attempt6 SUMO records; not additional samples',cases=selected),indent=2)+'\n',encoding='utf-8')


def render(source,out):
    out.mkdir(parents=True,exist_ok=True)
    if any((out/n).exists() for n in ('recorded_replay.gif','recorded_replay.png','replay_provenance.json')):
        raise FileExistsError('Keep previous media')
    data=json.loads(source.read_bytes());frames=[];poster=None
    try:
        font=ImageFont.load_default(size=18);small=ImageFont.load_default(size=15)
    except TypeError:  # Pillow 10.0, supported by the repository dependency range.
        font=small=ImageFont.load_default()
    colors=['#2674ad','#d07a19','#5a8a53']
    for kind in ('baseline_only','learning_adopted'):
        case=data['cases'][kind]
        sequences=[[dict(t=0,x=case['initial'],command=None,learning=False)]+case[k] for k in ('target','baseline')]
        last=min(len(v) for v in sequences)
        extent=max(80,max(v['x'][i][0] for sequence in sequences for v in sequence for i in range(3)))+10
        left=min(v[0] for v in case['initial'])-10
        def px(x):return int(185+(x-left)/(extent-left)*680)
        for index in range(last):
            a,b=(v[index] for v in sequences)
            if a['t']!=b['t']:raise ValueError('Only matched observed timestamps may be displayed')
            im=Image.new('RGB',(960,430),'#f8fafc');d=ImageDraw.Draw(im)
            d.text((28,18),'CoopGate | Recorded SUMO trajectory replay',font=font,fill='#142438')
            d.text((28,52),f"Case {case['case_id']} | actor 4703 | {case['scenario']} | {case['plant']}",font=small,fill='#43536a')
            status='LEARNED COMMAND' if a['learning'] else ('INITIAL STATE' if index==0 else 'BASELINE COMMAND')
            d.rounded_rectangle((28,84,930,119),radius=7,fill='#dcefe2' if a['learning'] else '#e5eaf0')
            d.text((43,92),f"t = {a['t']:.1f} s  |  last executed interval: {status}",font=small,fill='#142438')
            for mark,label in [(0,'merge reference'),(60,'finish reference')]:
                d.line((px(mark),151,px(mark),313),fill='#a8b3c1',width=2)
                d.text((px(mark)-40,133),label,font=small,fill='#536171')
            for role,name in enumerate(['Main-road front','Merging vehicle','Main-road rear']):
                y=180+role*53
                d.text((28,y-8),name,font=small,fill=colors[role]);d.line((185,y,905,y),fill='#d5dde6',width=2)
                bx=px(b['x'][role][0]);x=px(a['x'][role][0])
                d.ellipse((bx-10,y-10,bx+10,y+10),outline='#6f7783',width=2)
                d.rounded_rectangle((x-9,y-6,x+9,y+6),radius=3,fill=colors[role])
                d.text((185,y+13),f"s={a['x'][role][0]:6.1f} m   v={a['x'][role][1]:4.1f} m/s",font=small,fill='#43536a')
            d.text((28,341),'Solid: gated MAPPO     Hollow: paired traditional baseline     Playback: 2x',font=small,fill='#142438')
            d.text((28,369),'Recorded longitudinal progress; rows separate roles, not physical lane geometry.',font=small,fill='#536171')
            d.text((28,395),'Schematic replay, not SUMO-GUI footage. Two illustrative cases, not aggregate evidence.',font=small,fill='#536171')
            frames.append(im)
            if a['learning'] and poster is None:poster=im.copy()
        frames.extend([frames[-1]]*4)
    frames[0].save(out/'recorded_replay.gif',save_all=True,append_images=frames[1:],duration=250,loop=0,optimize=False)
    (poster or frames[0]).save(out/'recorded_replay.png')
    with Image.open(out/'recorded_replay.gif') as encoded:encoded_frames=encoded.n_frames
    proof=dict(source_extract=source.relative_to(ROOT).as_posix(),source_extract_sha256=sha(source),
        renderer_sha256=sha(Path(__file__)),rendered_frames=len(frames),encoded_frames=encoded_frames,fps=4,playback_speed=2,
        physical_lateral_geometry_rendered=False,new_simulations=0,new_statistical_inference=False,
        cases={kind:{k:v for k,v in case.items() if k not in ('initial','target','baseline')} for kind,case in data['cases'].items()},
        outputs={n:sha(out/n) for n in ('recorded_replay.gif','recorded_replay.png')})
    (out/'replay_provenance.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(frames=len(frames),gif_bytes=(out/'recorded_replay.gif').stat().st_size,cases=list(proof['cases']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extract',action='store_true');parser.add_argument('--out',type=Path,default=ROOT/'media')
    a=parser.parse_args();source=ROOT/'media/replay_cases.json'
    if a.extract:source.parent.mkdir(exist_ok=True);extract(source)
    render(source,a.out)
