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
    for name in ['C:/Windows/Fonts/msyh.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    raise RuntimeError('中文渲染需要微软雅黑、Noto Sans CJK 或文泉驿微米黑字体')


def render(source,out,stats):
    out.mkdir(parents=True,exist_ok=False)
    case=json.loads(source.read_bytes());full=json.loads(stats.read_bytes())
    methods=['baseline','nominal_4703','4703'];titles=['传统规划基线','同候选池名义方案','置信门控多智能体策略']
    notes=['作为对照的传统控制器','保留物理检查，去除置信与历史过滤','物理检查与逐车相对性能筛选']
    records=case['methods'];base=records['baseline']['row'];n=max(len(records[m]['trace']) for m in methods)
    assert all(records[m]['row']['initial']==case['initial'] for m in methods)
    assert all(records[m]['row'][key]==base[key] for m in methods for key in
               ['seed','scenario','plant','disturbance','innovation_tape_sha256'])
    story=case['story_type']
    assert story in ('fallback','benefit')
    gain=100*(base['team_cost']-records['4703']['row']['team_cost'])/base['team_cost']
    assert story!='benefit' or gain>=5
    def harmful(row):return any(row['per_vehicle_cost'][i]>base['per_vehicle_cost'][i]+1e-9 or row['per_vehicle_finish'][i]>base['per_vehicle_finish'][i]+1e-9 for i in range(3))
    assert not harmful(records['4703']['row'])
    if story=='fallback':
        assert records['4703']['row']['learned_action_steps']==0
        assert len(records['4703']['trace'])==len(records['baseline']['trace'])
        assert all(all(a[key]==b[key] for key in ['t','x','command']) for a,b in zip(records['4703']['trace'],records['baseline']['trace']))
    f={s:font(s) for s in [15,16,18,20,22,26,30]};frames=[];durations=[];poster=None;final=None;finish=None
    def txt(d,xy,text,size=18,color='#d9e4f1'):
        box=d.textbbox(xy,text,font=f[size])
        if box[2]>1272 or box[3]>899:raise ValueError('文字超出画面边界：'+text)
        d.text(xy,text,font=f[size],fill=color)
    for k in range(n):
        for sub in range(1,5):
            frac=sub/4;time=(k+frac)*.5;done=k==n-1 and sub==4
            im=Image.new('RGB',(1280,900),'#0d1726');d=ImageDraw.Draw(im)
            txt(d,(24,15),('案例一：全程回退，与传统基线一致' if story=='fallback' else '案例二：许可时采用学习策略，保留明显通行收益'),30,'#f5f9ff')
            txt(d,(24,57),f"相同初态与扰动｜确认案例 {case['case_id'].rsplit(':',1)[-1]}｜策略编号 4703｜时间 {time:4.2f} 秒",18)
            txt(d,(24,85),'本例用于说明门控行为；所有方案均保留物理检查。红色表示性能退化，不表示碰撞。',16,'#a9bbcf')
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
                txt(d,(merge[0]-23,y+96),'汇入点',15,'#f1cc70')
                for i,(s,v) in enumerate(positions):
                    if -125<=s<=95:
                        xx,yy=point(s,i)
                        d.rounded_rectangle((xx-11,yy-6,xx+11,yy+6),radius=3,fill=COLORS[i],outline='#edf5ff')
                        txt(d,(xx-6,yy-26),['前','汇','后'][i],16,COLORS[i])
                    txt(d,(x+13+i*130,y+211),f"{['前','汇','后'][i]}：{s:+.1f}米",15,COLORS[i])
                state=f"完成于 {trace[-1]['t']:.2f} 秒（保留最后记录）" if finished else ('采用学习指令' if used else ('执行传统基线' if m=='baseline' else '回退传统基线'))
                d.rounded_rectangle((x+12,y+239,x+394,y+270),radius=5,fill='#174e42' if used else '#283b53')
                txt(d,(x+23,y+245),state,16,'#65e4b7' if used else '#d5e4f3')
                txt(d,(x+14,y+283),'实际加速度［前车、汇入车、后车］（米/秒²）',15,'#a9bbcf')
                txt(d,(x+14,y+307),'  '.join(f'{value:+5.2f}' for value in cur['command']),22,'#ffffff')
                if m=='4703':
                    score=cur.get('selected_gain_confidence_score')
                    scoretext=('回合已结束，显示最后一条指令' if finished else
                        (f'通过门控：收益分数 {100*score:.3f}% ≥ 0.250%' if score is not None and used else '未选中学习指令，执行基线'))
                    txt(d,(x+14,y+344),scoretext,16,'#65e4b7' if used else '#a9bbcf')
                else:txt(d,(x+14,y+344),f"已采用学习指令的控制区间：{sum(bool(v['actual_learning_used']) for v in trace[:k+1])} 个",16,'#a9bbcf')
            d.rounded_rectangle((18,528,624,806),radius=10,fill='#17263a')
            txt(d,(34,542),'本例结果：逐车与传统基线比较' if done else '逐时刻门控记录：提案、基线与实际指令',20,'#ffffff')
            live=records['4703']['trace'][min(k,len(records['4703']['trace'])-1)]
            txt(d,(34,573),'最终逐车代价差（负值更优）' if done else f"记录区间 {live['t']-.5:.2f}—{live['t']:.2f} 秒；加速度单位：米/秒²",16,'#a9bbcf')
            if done:
                for i,label in enumerate(['前车','汇入车','后车']):
                    yy=610+i*46;txt(d,(36,yy),label,18,COLORS[i]);zero=360
                    d.line((zero,yy-2,zero,yy+35),fill='#a9bbcf',width=1)
                    for m,color,dy in [('nominal_4703','#ef8a78',0),('4703','#35c99c',19)]:
                        delta=records[m]['row']['per_vehicle_cost'][i]-base['per_vehicle_cost'][i]
                        maximum=max(abs(records[m]['row']['per_vehicle_cost'][j]-base['per_vehicle_cost'][j]) for m in ['nominal_4703','4703'] for j in range(3))
                        end=zero+delta*min(1100,150/max(maximum,1e-9))
                        if abs(delta)>1e-10:d.rectangle((min(zero,end),yy+dy,max(zero,end),yy+dy+12),fill=color)
                        txt(d,(470,yy+dy-4),f'{delta:+.4f}',15,color)
                dn=records['nominal_4703']['row']['per_vehicle_cost'][1]-base['per_vehicle_cost'][1]
                dg=records['4703']['row']['per_vehicle_cost'][1]-base['per_vehicle_cost'][1]
                earlier=base['per_vehicle_finish'][1]-records['4703']['row']['per_vehicle_finish'][1]
                txt(d,(34,755),f'门控：汇入车提前 {earlier:.1f} 秒；团队代价改善 {gain:.3f}%',16,'#ffffff')
                txt(d,(34,779),'代价＝到达时间＋0.05×实际指令总变化量',15,'#a9bbcf')
            else:
                for xx,label in [(36,'车辆'),(175,'原始学习提案'),(330,'同状态基线'),(475,'最终执行')]:txt(d,(xx,607),label,16,'#a9bbcf')
                for role,label in enumerate(['前车','汇入车','后车']):
                    yy=637+role*30;txt(d,(36,yy),label,18,COLORS[role])
                    for xx,key in [(195,'actor_proposal'),(350,'baseline_command'),(490,'command')]:
                        txt(d,(xx,yy),f"{live[key][role]:+.3f}",18,'#ffffff')
                score=live['selected_gain_confidence_score']
                msg=(f'收益分数 {100*score:.3f}% ≥ 0.250%；其他许可检查通过' if live['actual_learning_used'] else '无许可学习指令；分数记为“—”，不是零分。')
                txt(d,(36,746),msg,16,'#65e4b7' if live['actual_learning_used'] else '#f1cc70')
                txt(d,(36,777),'原始提案经候选构造与物理修复后，才可能被执行。',15,'#a9bbcf')
            d.rounded_rectangle((640,528,1262,806),radius=10,fill='#17263a')
            txt(d,(658,542),'完整确认集：1,024 个独立初态',20,'#ffffff')
            txt(d,(658,575),'有退化的采用回合数／采用回合数',16,'#a9bbcf')
            for j,(m,name,color) in enumerate([('4703','置信门控','#35c99c'),('nominal_4703','名义方案','#ef8a78'),('thinned_4703','随机弃权','#a9bbcf')]):
                q=full['confirmation_methods'][m];yy=613+j*43;rate=q['harmful_adopted']/q['adopted']
                txt(d,(659,yy),name,18,color);d.rectangle((770,yy+5,770+rate*2100,yy+20),fill=color)
                txt(d,(1020,yy),f"{q['harmful_adopted']}/{q['adopted']}",18,color)
                txt(d,(1130,yy),f'{100*rate:.1f}%',18,color)
            means=[full['confirmation_methods'][m]['mean_paired_percentage_gain'] for m in ['4703','nominal_4703','thinned_4703']]
            txt(d,(658,750),f'平均收益：门控 {means[0]:.3f}%｜名义 {means[1]:.3f}%｜弃权 {means[2]:.3f}%',15)
            txt(d,(658,777),'以上为描述性结果；随机弃权不保证采用率相等。',15,'#a9bbcf')
            txt(d,(24,824),'前／汇／后分别表示前车、汇入车和后车。道路投影和帧间线性插值仅用于示意。',16,'#a9bbcf')
            txt(d,(24,850),'数据来自封存的仿真记录；正常片段 2 倍速，采用节点暂停。非软件界面录屏，不新增实验。',16,'#a9bbcf')
            txt(d,(24,876),'本例为按结果条件选取的说明案例；整体结论以完整实验为准。性能退化不等于碰撞，零观测不代表零风险。',15,'#f1cc70')
            frames.append(im);durations.append([60,60,60,70][sub-1])
            if adopted_now and sub==4 and poster is None:
                poster=im.copy();durations[-1]+=1800
            if story=='benefit' and k==len(records['4703']['trace'])-1 and sub==4:
                finish=im.copy();durations[-1]+=1800
            if done:final=im.copy()
    assert final is not None
    frames.append(final);durations.append(4000)
    # One palette keeps static pixels stable across frames and permits compact
    # GIF delta encoding, without changing the recorded numerical data.
    palette=frames[0].quantize(colors=128)
    encoded=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    encoded[0].save(out/'comparison.gif',save_all=True,append_images=encoded[1:],duration=durations,loop=0,optimize=False)
    final.save(out/'comparison.png');(poster or final).save(out/'adoption.png')
    (finish or final).save(out/'finish.png')
    with Image.open(out/'comparison.gif') as opened:count=opened.n_frames
    proof=dict(case_id=case['case_id'],selection=case['selection'],new_simulations=0,new_inference=False,
        schematic_road=True,linear_interpolation_for_display_only=True,playback_speed=2,adoption_pause_ms=1800 if story=='benefit' else 0,
        completion_pause_ms=1800 if story=='benefit' else 0,final_hold_ms=4000,
        story_type=story,source_case_sha256=sha(source),source_statistics=stats.relative_to(ROOT).as_posix(),source_statistics_sha256=sha(stats),
        renderer_sha256=sha(Path(__file__)),encoded_frames=count,display_language='zh-CN',
        outputs={n:sha(out/n) for n in ['comparison.gif','comparison.png','adoption.png','finish.png']})
    (out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(encoded_frames=count,gif_bytes=(out/'comparison.gif').stat().st_size,case=case['case_id'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--case',choices=['fallback','benefit'],required=True);a=p.parse_args()
    render(ROOT/f'media/gate_story/{a.case}.json',a.out,ROOT/'artifacts/mappo_ctde_20261101/formal_mappo_attempt6_top_two_recovery_20261104/analysis/comparisons.json')
