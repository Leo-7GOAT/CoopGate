"""Build a genuine MAPPO manuscript only from the TERMINAL complete formal study.

Use bundled artifact Python. Run the artifact-start marker before invocation.
The old PI manuscript and every original artifact remain unchanged.
Rendering and visual review are mandatory before Desktop delivery.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from build_completed_manuscript_20261101 import run_math,scripted,fraction,summation

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101'
FORMAL=BASE/'formal_mappo_attempt3_20261102'
REFERENCE=ROOT/'artifacts/paper_supplement_20261026/CoopGate_论文实验与理论补充_20261026.docx'
REFERENCE_SHA='d4310c3a98cae08438cb46db8f8414710c41b2f4d57c39f82fc386b170729ec6'
INPUTS={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    INPUTS[Path(p).relative_to(ROOT).as_posix()]=sha(p)
    return json.loads(Path(p).read_bytes())


def evidence():
    # This gate precedes ALL access to scientific outcomes,including past ones.
    receipt=read(FORMAL/'completion_receipt.json')
    if (receipt['actual_native_finalizer']['actual_native_exit_code']!=0
        or not receipt['all_collector_and_reviewer_native_exits_zero']
        or receipt['main_actual_sumo_runs']!=16384 or receipt['total_actual_sumo_runs']!=37888
        or receipt['independent_initials']!=4096):raise ValueError('Full actual terminal formal evidence required')
    stages=receipt['stages']
    expected={(s,a,i) for s in ('calibration','confirmation') for a in ('collect','review') for i in range(4)}
    if len(stages)!=16 or {(e['stage'],e['action'],e['shard']) for e in stages}!=expected or any(e['actual_native_exit_code']!=0 for e in stages):
        raise ValueError('Every actual collector/reviewer must have exited zero')
    for e in stages:
        directory=FORMAL/e['stage']/f"shard{e['shard']}"
        name='verification.json' if e['action']=='review' else 'collection_complete.json'
        key='verification_sha256' if e['action']=='review' else 'collection_complete_sha256'
        if sha(directory/name)!=e[key]:raise ValueError('Native-reviewed data changed')
        proof=read(directory/name)
        expected_n=(5376 if e['stage']=='calibration' else 4096)
        if proof['executions']!=expected_n:raise ValueError('Full stage scope changed')
    for name in ('certificate','comparisons'):
        if sha(FORMAL/f'analysis/{name}.json')!=receipt[name+'_sha256']:raise ValueError('Finalized analysis changed')
    cert=read(FORMAL/'analysis/certificate.json');comparison=read(FORMAL/'analysis/comparisons.json')
    if cert['fixed_models']!=[4703,4704,4705] or cert['common_gain_floor']!=.0025 or not cert['all_formal_thresholds_preserved']:
        raise ValueError('Wrong algorithm family or relaxed scientific thresholds')
    audits=[]
    for seed in (4703,4704,4705):
        audit=read(BASE/f'development/joint_{seed}/audit.json')
        if not (audit['algorithm']=='mappo_ctde_20261101' and audit['steps']==2097152 and audit['final_budget_complete']
            and audit['on_policy_proposals_verified'] and audit['independent_gae_verified']):raise ValueError('Genuine joint PPO audit missing')
        audits.append(audit)
    competence=read(BASE/'development/single_4601_sumo_full/single_competence.json')
    return cert,comparison,competence


def build(out):
    if out.exists():raise FileExistsError('Keep every manuscript revision')
    cert,comparison,competence=evidence()
    if sha(REFERENCE)!=REFERENCE_SHA:raise ValueError('Retained layout reference changed')
    doc=Document(REFERENCE);section=doc.sections[0]._sectPr.xml
    for child in list(doc.element.body):
        if child.tag!=qn('w:sectPr'):doc.element.body.remove(child)
    doc.core_properties.title='基于性能门控的多智能体强化学习与传统控制协同方法'
    doc.core_properties.author='李东旭';doc.core_properties.subject='真实 MAPPO 联合训练与独立闭环验证'
    doc.core_properties.version='20261102'
    def p(text,style=None):
        node=doc.add_paragraph(text,style);node.paragraph_format.first_line_indent=Cm(0 if style else .74)
        if not style:node.paragraph_format.space_after=Pt(5)
        return node
    def h(text,level=1):return p(text,'Heading '+str(level))
    def eq(*nodes):
        para=p('');para.alignment=WD_ALIGN_PARAGRAPH.CENTER;para.paragraph_format.first_line_indent=Cm(0)
        para.paragraph_format.keep_together=True;math=OxmlElement('m:oMath')
        for n in nodes:math.append(run_math(n) if isinstance(n,str) else n)
        para._p.append(math)
    def table(caption,headers,rows,widths):
        cap=p(caption,'Caption');cap.paragraph_format.keep_with_next=True
        t=doc.add_table(rows=1,cols=len(headers));t.autofit=False
        for column,width in zip(t.columns,widths):column.width=Cm(width)
        for cell,value in zip(t.rows[0].cells,headers):cell.text=str(value)
        for row in rows:
            for cell,value in zip(t.add_row().cells,row):cell.text=str(value)
        borders=OxmlElement('w:tblBorders')
        for edge in ('top','left','bottom','right','insideH','insideV'):
            node=OxmlElement('w:'+edge)
            for k,v in dict(val='single',sz='4',color='D9D9D9').items():node.set(qn('w:'+k),v)
            borders.append(node)
        t._tbl.tblPr.append(borders)
        for ri,row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            if ri==0:row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            for ci,(cell,width) in enumerate(zip(row.cells,widths)):
                cell.width=Cm(width);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                margins=OxmlElement('w:tcMar')
                for edge in ('top','left','bottom','right'):
                    n=OxmlElement('w:'+edge);n.set(qn('w:w'),'80');n.set(qn('w:type'),'dxa');margins.append(n)
                cell._tc.get_or_add_tcPr().append(margins)
                if ri==0:
                    fill=OxmlElement('w:shd');fill.set(qn('w:fill'),'DCE6F1');cell._tc.get_or_add_tcPr().append(fill)
                for para in cell.paragraphs:
                    para.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                    para.paragraph_format.first_line_indent=Cm(0);para.paragraph_format.space_after=Pt(3)
                    para.paragraph_format.space_before=Pt(3);para.paragraph_format.line_spacing=1.05
                    if ri==0 or (len(t.rows)<=6 and ri<len(t.rows)-1):para.paragraph_format.keep_with_next=True
                    for run in para.runs:run.font.size=Pt(10);run.bold=ri==0
        p('')
    def num(x,d=3):return '不可评估' if x is None else f'{x:.{d}f}'
    from mappo_manuscript_text_20261102 import compose
    compose(p,h,eq,table,num,scripted,fraction,summation,cert,comparison,competence)
    if doc.sections[0]._sectPr.xml!=section:raise ValueError('Reference geometry changed')
    for name,digest in INPUTS.items():
        if sha(ROOT/name)!=digest:raise ValueError('Completed evidence changed during authoring')
    out.mkdir(parents=True,exist_ok=False);target=out/'CoopGate_真实MAPPO论文初稿_20261102.docx';doc.save(target)
    with ZipFile(REFERENCE) as original,ZipFile(target) as final:
        if set(original.namelist())!=set(final.namelist()):raise ValueError('Unexpected template component change')
        for name in original.namelist():
            if name not in ('word/document.xml','docProps/core.xml') and original.read(name)!=final.read(name):
                raise ValueError('Preserve-only template component changed: '+name)
    record=dict(document_sha256=sha(target),reference_sha256=REFERENCE_SHA,completed_inputs_sha256=INPUTS,
        builder_sha256=sha(__file__),text_source_sha256=sha(ROOT/'operations/mappo_manuscript_text_20261102.py'),
        native_equations=len(list(doc.element.body.iter(qn('m:oMath')))),live_partial_outcomes_read=False,
        old_pi_manuscript_unchanged=True,template_opaque_parts_byte_preserved=True,final_visual_review_required=True)
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(target),native_equations=record['native_equations']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.out)
