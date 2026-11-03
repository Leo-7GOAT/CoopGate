# Layout derived without changing original builder sha256=f7782c791d22004205d7c5246cbd08a67b86a9f947f339952cdf79980dc9b555
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
from lxml import etree
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from build_completed_manuscript_20261101 import run_math,scripted,fraction,summation

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101'
FORMAL=BASE/'formal_mappo_attempt5_powered_20261103'
REFERENCE=ROOT/'artifacts/paper_supplement_20261026/CoopGate_论文实验与理论补充_20261026.docx'
REFERENCE_SHA='d4310c3a98cae08438cb46db8f8414710c41b2f4d57c39f82fc386b170729ec6'
FIGURES=BASE/'formal_mappo_attempt5_powered_figures_20261103'
INPUTS={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    INPUTS[Path(p).relative_to(ROOT).as_posix()]=sha(p)
    return json.loads(Path(p).read_bytes())


def evidence():
    bundle=EVIDENCE_BUNDLE
    terminal_path=bundle.with_suffix('.native.json')
    terminal=read(terminal_path)
    if terminal['actual_native_exit_code']!=0 or terminal['bundle_sha256']!=sha(bundle):
        raise ValueError('Actual completed evidence export required before authoring')
    data=read(bundle)
    if data['exporter_sha256']!=sha(ROOT/'operations/build_powered_history_evidence_20261103.py'):
        raise ValueError('Completed evidence exporter changed')
    for name,digest in data['inputs_sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path)!=digest:
            raise ValueError('Completed input changed: '+name)
        INPUTS[name]=digest
    owner=read(FORMAL/'orchestrator_native_receipt.json')
    if owner['actual_native_exit_code']!=0 or owner['completion_receipt_sha256']!=sha(FORMAL/'completion_receipt.json'):
        raise ValueError('Actual primary native terminal required')
    cert=read(FORMAL/'analysis/certificate.json');comparison=read(FORMAL/'analysis/comparisons.json')
    if cert!=data['certificate'] or comparison!=data['comparison']:
        raise ValueError('Bundle differs from completed primary evidence')
    if (cert['attempt']!=5 or cert['allocated_error']!=.05/30
            or cert['calibration_initials']!=4096 or cert['confirmation_initials']!=1024
            or not data['pipeline']['completed']
            or data['pipeline']['supplementary_actual_sumo_executions']!=6144):
        raise ValueError('Full unchanged study and registered ablations required')
    return cert,comparison,data

def build(out):
    if out.exists():raise FileExistsError('Keep every manuscript revision')
    cert,comparison,competence=evidence()
    figures=read(FIGURES/'report.json')
    if (figures['source_certificate_sha256']!=sha(FORMAL/'analysis/certificate.json')
        or figures['source_comparisons_sha256']!=sha(FORMAL/'analysis/comparisons.json')
        or figures['owner_receipt_sha256']!=sha(FORMAL/'orchestrator_native_receipt.json')):
        raise ValueError('Figures do not describe this exact completed formal study')
    image_hashes={x['path']:x['sha256'] for x in figures['figures'] if x['path'].endswith('.png')}
    for name,digest in image_hashes.items():
        if sha(FIGURES/name)!=digest:raise ValueError('Formal figure changed')
    if sha(REFERENCE)!=REFERENCE_SHA:raise ValueError('Retained layout reference changed')
    doc=Document(REFERENCE);section=doc.sections[0]._sectPr.xml
    for child in list(doc.element.body):
        if child.tag!=qn('w:sectPr'):doc.element.body.remove(child)
    doc.core_properties.title='基于观测历史与相对性能门控的多车学习控制协同方法'
    doc.core_properties.author='李东旭';doc.core_properties.subject='真实 MAPPO 联合训练与独立闭环验证'
    doc.core_properties.version='20261103'
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
    def figure(name,caption):
        if name+'.png' not in image_hashes:raise ValueError('Required completed figure missing')
        para=p('');para.paragraph_format.keep_with_next=True;para.paragraph_format.first_line_indent=Cm(0)
        para.alignment=WD_ALIGN_PARAGRAPH.CENTER;para.add_run().add_picture(str(FIGURES/(name+'.png')),width=Cm(16.4))
        cap=p(caption,'Caption');cap.paragraph_format.keep_with_next=False
        cap.paragraph_format.keep_together=True
    from powered_mappo_manuscript_text_20261103 import compose
    compose(p,h,eq,table,num,scripted,fraction,summation,cert,comparison,competence,figure)
    if len(doc.inline_shapes)!=3:raise ValueError('Exactly three complete scientific figures required')
    if doc.sections[0]._sectPr.xml!=section:raise ValueError('Reference geometry changed')
    for name,digest in INPUTS.items():
        if sha(ROOT/name)!=digest:raise ValueError('Completed evidence changed during authoring')
    out.mkdir(parents=True,exist_ok=False);target=out/'CoopGate_真实MAPPO论文完整实验版_20261103.docx';doc.save(target)
    with ZipFile(REFERENCE) as original,ZipFile(target) as final:
        old=set(original.namelist());new=set(final.namelist())
        if not old<=new:raise ValueError('Original template component removed')
        for name in new-old:
            if not name.startswith('word/media/') or hashlib.sha256(final.read(name)).hexdigest() not in image_hashes.values():
                raise ValueError('Unexpected new package component')
        for name in original.namelist():
            if name not in ('word/document.xml','docProps/core.xml','[Content_Types].xml','word/_rels/document.xml.rels') and original.read(name)!=final.read(name):
                raise ValueError('Preserve-only template component changed: '+name)
        def relationships(bundle):
            return {n.get('Id'):dict(n.attrib) for n in etree.fromstring(bundle.read('word/_rels/document.xml.rels'))}
        oldrels,newrels=relationships(original),relationships(final)
        if any(newrels.get(k)!=v for k,v in oldrels.items()):raise ValueError('Existing template relationship changed')
    record=dict(document_sha256=sha(target),reference_sha256=REFERENCE_SHA,completed_inputs_sha256=INPUTS,
        builder_sha256=sha(__file__),text_source_sha256=sha(ROOT/'operations/powered_mappo_manuscript_text_20261103.py'),
        native_equations=len(list(doc.element.body.iter(qn('m:oMath')))),live_partial_outcomes_read=False,
        embedded_completed_figures=len(doc.inline_shapes),existing_template_relationships_preserved=True,
        old_pi_manuscript_unchanged=True,template_opaque_parts_byte_preserved=True,final_visual_review_required=True)
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(target),native_equations=record['native_equations']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();EVIDENCE_BUNDLE=a.evidence;build(a.out)
