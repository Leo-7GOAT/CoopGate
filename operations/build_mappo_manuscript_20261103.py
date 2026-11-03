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
FORMAL=BASE/'formal_mappo_attempt4_20261102'
REFERENCE=ROOT/'artifacts/paper_supplement_20261026/CoopGate_论文实验与理论补充_20261026.docx'
REFERENCE_SHA='d4310c3a98cae08438cb46db8f8414710c41b2f4d57c39f82fc386b170729ec6'
FIGURES=BASE/'formal_mappo_attempt4_figures_20261102'
INPUTS={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    INPUTS[Path(p).relative_to(ROOT).as_posix()]=sha(p)
    return json.loads(Path(p).read_bytes())


def evidence():
    # This gate precedes ALL access to scientific outcomes,including past ones.
    receipt=read(FORMAL/'completion_receipt.json')
    owner=read(FORMAL/'orchestrator_native_receipt.json')
    if owner['actual_native_exit_code']!=0 or owner['completion_receipt_sha256']!=sha(FORMAL/'completion_receipt.json'):
        raise ValueError('Observed native owner exit for this exact completion required')
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
    plan=read(FORMAL/'plan.json');allocation=read(FORMAL/'allocation.json')
    if allocation['attempt']!=4 or allocation['allocated_error']!=.05/20 or cert['allocated_error']!=.05/20:
        raise ValueError('Wrong spent registration/error allocation')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen implementation changed: '+name)
    if cert['fixed_models']!=[4703,4704,4705] or cert['common_gain_floor']!=.0025 or not cert['all_formal_thresholds_preserved']:
        raise ValueError('Wrong algorithm family or relaxed scientific thresholds')
    audits=[]
    for seed in (4703,4704,4705):
        audit=read(BASE/f'development/joint_{seed}/audit.json')
        if not (audit['algorithm']=='mappo_ctde_20261101' and audit['steps']==2097152 and audit['final_budget_complete']
            and audit['on_policy_proposals_verified'] and audit['independent_gae_verified']):raise ValueError('Genuine joint PPO audit missing')
        audits.append(audit)
    competence=read(BASE/'development/single_4601_sumo_full/single_competence.json')
    read(BASE/'development/single_4603_exact/single_competence.json')
    read(BASE/'ippo_matched_family_20261102.json')
    development=read(BASE/'gain_margin_fresh_analysis_20261102/independent_audit.json')
    if development['actual_sumo_executions']!=1040 or development['independent_initials']!=80:
        raise ValueError('Complete independent development evidence required')
    diagnostic=read(BASE/'parameter_counterexample_diagnosis_20261102.json')
    if (not diagnostic['privileged_offline_only'] or diagnostic['online_controller_changed']
        or diagnostic['independent_adverse_initials']!=1
        or any(e['telescoping_identity_residual_max']!=0 for e in diagnostic['examples'])):
        raise ValueError('Offline counterexample evidence changed')
    paired=BASE/'coupled_holdout_ablation_recovery_20261103'
    paired_owner=read(paired/'owner_native_exit.json');paired_report=read(paired/'report.json')
    if paired_owner['actual_native_exit_code']!=0 or paired_owner['report_sha256']!=sha(paired/'report.json'):
        raise ValueError('Complete paired recovery native owner required')
    if not paired_report['completed'] or paired_report['complete_recorded_old_gate_outcomes']!=3072 or paired_report['actual_resumed_sumo_executions']!=1697:
        raise ValueError('Full registered old/new comparison required')
    if paired_report['primary_certificate_sha256']!=sha(FORMAL/'analysis/certificate.json'):
        raise ValueError('Paired comparison describes another study')
    read(BASE/'formal_mappo_attempt4_diagnosis_20261102/report.json')
    adverse=read(BASE/'formal_mappo_attempt4_diagnosis_20261102/all_false_admissions.json')
    if adverse['method_initial_incidents']!=18 or adverse['independent_adverse_initials']!=12:
        raise ValueError('Incomplete retained adverse cohort')
    competence['_paired_recovery']=paired_report
    development=BASE/'development/observed_history_validation_20261103'
    development_owner=read(development/'owner_native_exit.json');development_report=read(development/'report.json')
    if (development_owner['actual_native_exit_code']!=0 or development_owner['report_sha256']!=sha(development/'report.json')
        or not development_report['completed'] or development_report['actual_sumo_executions']!=644
        or development_report['new_independent_development_initials']!=80 or development_report['known_regression_initials']!=12
        or any(c['actual_native_exit_code']!=0 for c in development_report['children'])):
        raise ValueError('Complete new development study required')
    development_plan=read(development/'plan.json')
    for name,digest in development_plan['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Development source changed')
    competence['_history_development']=development_report
    return cert,comparison,competence


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
    doc.core_properties.title='基于性能门控的多智能体强化学习与传统控制协同方法'
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
    from mappo_manuscript_text_20261103 import compose
    compose(p,h,eq,table,num,scripted,fraction,summation,cert,comparison,competence,figure)
    if len(doc.inline_shapes)!=3:raise ValueError('Exactly three complete scientific figures required')
    if doc.sections[0]._sectPr.xml!=section:raise ValueError('Reference geometry changed')
    for name,digest in INPUTS.items():
        if sha(ROOT/name)!=digest:raise ValueError('Completed evidence changed during authoring')
    out.mkdir(parents=True,exist_ok=False);target=out/'CoopGate_真实MAPPO论文初稿_20261103.docx';doc.save(target)
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
        builder_sha256=sha(__file__),text_source_sha256=sha(ROOT/'operations/mappo_manuscript_text_20261103.py'),
        native_equations=len(list(doc.element.body.iter(qn('m:oMath')))),live_partial_outcomes_read=False,
        embedded_completed_figures=len(doc.inline_shapes),existing_template_relationships_preserved=True,
        old_pi_manuscript_unchanged=True,template_opaque_parts_byte_preserved=True,final_visual_review_required=True)
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(target),native_equations=record['native_equations']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.out)
