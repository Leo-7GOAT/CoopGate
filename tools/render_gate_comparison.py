"""Three-method explanation from a fixed, disclosed archived SUMO case.

Road projection and between-sample interpolation are presentation only. Full
confirmation statistics come from the unchanged original comparison file.
No policy, simulation, confidence test or experimental outcome is recomputed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
COLORS=['#3887d6','#ee9b35','#a482df']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def font(size):
    for name in ['DejaVuSans.ttf','C:/Windows/Fonts/segoeui.ttf']:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    try:return ImageFont.load_default(size=size)
    except TypeError:return ImageFont.load_default()


def render(source,out,stats):
    out.mkdir(parents=True,exist_ok=False)
    case=json.loads(source.read_bytes());full=json.loads(stats.read_bytes())
    methods=['baseline','nominal_4703','4703'];titles=['Traditional baseline','Same-pool nominal','Confidence-gated MAPPO']
    notes=['Reference controller','Physical checks; no confidence filter','Physical + relative-performance checks']
    records=case['methods'];base=records['baseline']['row'];n=max(len(records[m]['trace']) for m in methods)
    assert all(records[m]['row']['initial']==case['initial'] for m in methods)
    assert all(records[m]['row'][key]==base[key] for m in methods for key in
               ['seed','scenario','plant','disturbance','innovation_tape_sha256'])
    assert case['case_id']=='mappo6:confirmation:0160' and case['seed']==901000161
    assert records['4703']['row']['learned_action_steps']>0
    def harmful(row):return any(row['per_vehicle_cost'][i]>base['per_vehicle_cost'][i]+1e-9 or row['per_vehicle_finish'][i]>base['per_vehicle_finish'][i]+1e-9 for i in range(3))
    assert harmful(records['nominal_4703']['row']) and not harmful(records['4703']['row'])
    f={s:font(s) for s in [15,16,18,20,22,26,30]};frames=[];durations=[];poster=None;final=None
    def txt(d,xy,text,size=18,color='#d9e4f1'):d.text(xy,text,font=f[size],fill=color)
    for k in range(n):
        for sub in range(1,5):
            frac=sub/4;time=(k+frac)*.5;done=k==n-1 and sub==4
            im=Image.new('RGB',(1280,900),'#0d1726');d=ImageDraw.Draw(im)
            txt(d,(24,15),'WHEN SHOULD A LEARNED COMMAND BE USED?',30,'#f5f9ff')
            txt(d,(24,57),f"Same initial state + disturbance | {case['case_id']} | actor 4703 | t={time:4.2f}s",18)
            txt(d,(24,85),'Illustrative selected case. All methods retain physical checks. Red means performance regression, not collision.',16,'#a9bbcf')
            adopted_now=False
            for column,m in enumerate(methods):
                x=18+column*422;y=121;record=records[m];trace=record['trace'];finished=k>=len(trace) or (k==len(trace)-1 and sub==4)
                cur=trace[min(k,len(trace)-1)];prev=case['initial'] if k==0 else trace[min(k-1,len(trace)-1)]['x']
                positions=cur['x'] if finished else [[prev[i][j]+frac*(cur['x'][i][j]-prev[i][j]) for j in range(2)] for i in range(3)]
                used=bool(cur['actual_learning_used']) and not finished
                adopted_now |= m=='4703' and used
                border='#35c99c' if m=='4703' else ('#ef8a78' if m.startswith('nominal') else '#6a8cae')
                d.rounded_rectangle((x,y,x+406,y+389),radius=12,fill='#17263a',outline=border,width=2)
                txt(d,(x+14,y+12),titles[column],22,'#ffffff');txt(d,(x+14,y+44),notes[column],15,'#a9bbcf')
                # Schematic two-to-one road, using recorded route progress.
                def point(s,role):return (x+228+s*1.45,y+174+(0 if s>=0 else (1 if role!=1 else -1)*(-s)*.29))
                merge=(x+228,y+174)
                for role in [0,1]:d.line([point(-125,role),merge,point(95,role)],fill='#34455b',width=26)
                for role in [0,1]:d.line([point(-125,role),merge,point(95,role)],fill='#a2afc0',width=1)
                d.line((merge[0],y+124,merge[0],y+222),fill='#f1cc70',width=2)
                txt(d,(merge[0]-23,y+96),'merge',15,'#f1cc70')
                for i,(s,v) in enumerate(positions):
                    if -125<=s<=95:
                        xx,yy=point(s,i)
                        d.rounded_rectangle((xx-11,yy-6,xx+11,yy+6),radius=3,fill=COLORS[i],outline='#edf5ff')
                        txt(d,(xx-6,yy-26),['F','M','R'][i],16,COLORS[i])
                    txt(d,(x+13+i*130,y+211),f"{['F','M','R'][i]}: {s:+.1f}m",15,COLORS[i])
                state='COMPLETE (last recorded state)' if finished else ('LEARNED COMMAND' if used else ('BASELINE' if m=='baseline' else 'BASELINE / FALLBACK'))
                d.rounded_rectangle((x+12,y+239,x+394,y+270),radius=5,fill='#174e42' if used else '#283b53')
                txt(d,(x+23,y+245),state,16,'#65e4b7' if used else '#d5e4f3')
                txt(d,(x+14,y+283),'Executed acceleration [F, M, R] (m/s2)',15,'#a9bbcf')
                txt(d,(x+14,y+307),'  '.join(f'{value:+5.2f}' for value in cur['command']),22,'#ffffff')
                if m=='4703':
                    score=cur.get('selected_gain_confidence_score')
                    scoretext=('Episode ended; last recorded command shown' if finished else
                        (f'Accepted score {100*score:.3f}% >= 0.250%' if score is not None and used else 'No learned command selected; use baseline'))
                    txt(d,(x+14,y+344),scoretext,16,'#65e4b7' if used else '#a9bbcf')
                else:txt(d,(x+14,y+344),f"Adopted intervals so far: {sum(bool(v['actual_learning_used']) for v in trace[:k+1])}",16,'#a9bbcf')
            d.rounded_rectangle((18,528,624,806),radius=10,fill='#17263a')
            txt(d,(34,542),'CASE OUTCOME: each car vs its baseline',20,'#ffffff')
            txt(d,(34,573),'Final cost difference J_i (negative is better)',16,'#a9bbcf')
            if done:
                for i,label in enumerate(['Front','Merge','Rear']):
                    yy=610+i*46;txt(d,(36,yy),label,18,COLORS[i]);zero=360
                    d.line((zero,yy-2,zero,yy+35),fill='#a9bbcf',width=1)
                    for m,color,dy in [('nominal_4703','#ef8a78',0),('4703','#35c99c',19)]:
                        delta=records[m]['row']['per_vehicle_cost'][i]-base['per_vehicle_cost'][i]
                        end=zero+delta*1100
                        if abs(delta)>1e-10:d.rectangle((min(zero,end),yy+dy,max(zero,end),yy+dy+12),fill=color)
                        txt(d,(470,yy+dy-4),f'{delta:+.4f}',15,color)
                dn=records['nominal_4703']['row']['per_vehicle_cost'][1]-base['per_vehicle_cost'][1]
                dg=records['4703']['row']['per_vehicle_cost'][1]-base['per_vehicle_cost'][1]
                txt(d,(34,755),f'Merge-car cost: nominal {dn:+.4f} | gate {dg:+.4f}',16,'#ffffff')
                txt(d,(34,779),'J = arrival time + 0.05 x executed-command variation',15,'#a9bbcf')
            else:
                txt(d,(36,625),'Outcome revealed after the full recorded episode.',18)
                txt(d,(36,663),'Watch the green panel accept a candidate, then',18,'#a9bbcf')
                txt(d,(36,693),'continue on the baseline when none is selected.',18,'#a9bbcf')
                txt(d,(36,753),'Gate score is a prediction score, not a probability.',16,'#f1cc70')
            d.rounded_rectangle((640,528,1262,806),radius=10,fill='#17263a')
            txt(d,(658,542),'FULL CONFIRMATION: 1,024 initial states',20,'#ffffff')
            txt(d,(658,575),'Harmful adopted episodes / adopted episodes',16,'#a9bbcf')
            for j,(m,name,color) in enumerate([('4703','Gate','#35c99c'),('nominal_4703','Nominal','#ef8a78'),('thinned_4703','Thinned','#a9bbcf')]):
                q=full['confirmation_methods'][m];yy=613+j*43;rate=q['harmful_adopted']/q['adopted']
                txt(d,(659,yy),name,18,color);d.rectangle((770,yy+5,770+rate*2100,yy+20),fill=color)
                txt(d,(1020,yy),f"{q['harmful_adopted']}/{q['adopted']}",18,color)
                txt(d,(1130,yy),f'{100*rate:.1f}%',18,color)
            means=[full['confirmation_methods'][m]['mean_paired_percentage_gain'] for m in ['4703','nominal_4703','thinned_4703']]
            txt(d,(658,750),f'Mean gain: gate {means[0]:.3f}% | nominal {means[1]:.3f}% | thinned {means[2]:.3f}%',15)
            txt(d,(658,777),'Descriptive points; thinning does not prove equal adoption.',15,'#a9bbcf')
            txt(d,(24,824),'F / M / R = front / merging / rear vehicle. Road projection and linear interpolation are schematic.',16,'#a9bbcf')
            txt(d,(24,850),'Archived SUMO data; 2x playback with a pause at adoption. Not GUI footage. No new experiment.',16,'#a9bbcf')
            txt(d,(24,876),'Case chosen to illustrate the gate; population evidence is the full study. Performance harm is NOT collision risk.',15,'#f1cc70')
            frames.append(im);durations.append([60,60,60,70][sub-1])
            if adopted_now and sub==4 and poster is None:
                poster=im.copy();durations[-1]+=1800
            if done:final=im.copy()
    assert final is not None
    frames.append(final);durations.append(4000)
    # One palette keeps static pixels stable across frames and permits compact
    # GIF delta encoding, without changing the recorded numerical data.
    palette=frames[0].quantize(colors=128)
    encoded=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    encoded[0].save(out/'comparison.gif',save_all=True,append_images=encoded[1:],duration=durations,loop=0,optimize=False)
    final.save(out/'comparison.png');(poster or final).save(out/'adoption.png')
    with Image.open(out/'comparison.gif') as opened:count=opened.n_frames
    proof=dict(case_id=case['case_id'],selection=case['selection'],new_simulations=0,new_inference=False,
        schematic_road=True,linear_interpolation_for_display_only=True,playback_speed=2,adoption_pause_ms=1800,final_hold_ms=4000,
        source_case_sha256=sha(source),source_statistics=stats.relative_to(ROOT).as_posix(),source_statistics_sha256=sha(stats),
        renderer_sha256=sha(Path(__file__)),encoded_frames=count,
        outputs={n:sha(out/n) for n in ['comparison.gif','comparison.png','adoption.png']})
    (out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(encoded_frames=count,gif_bytes=(out/'comparison.gif').stat().st_size,case=case['case_id'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'runs/gate_comparison_render_20261105');a=p.parse_args()
    render(ROOT/'media/gate_comparison/case.json',a.out,ROOT/'artifacts/mappo_ctde_20261101/formal_mappo_attempt6_top_two_recovery_20261104/analysis/comparisons.json')
