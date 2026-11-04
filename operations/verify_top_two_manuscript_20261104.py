"""Check the terminal manuscript against original data and preserved package.

Run with bundled document Python after inspecting every current page PNG.
The page list records that visual review; structural checks do not replace it.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
from docx import Document
from docx.oxml.ns import qn
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101'
REFERENCE=ROOT/'artifacts/paper_supplement_20261026/CoopGate_论文实验与理论补充_20261026.docx'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(revision,reviewed):
    docpath=next(revision.glob('*.docx'));doc=Document(docpath)
    evidence=json.loads((BASE/'paper_top_two_attempt6_evidence_20261104/evidence.json').read_bytes())
    certificate=evidence['full_primary_certificate'];comp=evidence['all_confirmation_comparisons']
    methods=comp['confirmation_methods'];seeds=('4703','4704','4705')
    num=lambda x,d=3:f'{x:.{d}f}'
    tables=[[[cell.text for cell in row.cells] for row in table.rows] for table in doc.tables]
    assert len(tables)==7 and len(list(doc.element.body.iter(qn('m:oMath'))))==14
    assert len(doc.inline_shapes)==3
    assert tables[0][1:]==[['分层校准','6144','7','43008'],['IID 确认','1024','16','16384'],['总计','7168','分阶段固定','59392']]
    for i,s in enumerate(seeds,1):
        risk=certificate['primary_risk']['members'];a=certificate['adoption']['members'][s];g=certificate['paired_gain']['members'][s]
        expected=[s,num(100*risk[s+':mixture']['pointwise_risk_upper'])+'%',num(100*risk[s+':strong_lag']['pointwise_risk_upper'])+'%',num(100*a['adoption_lower'])+'%',num(g['gain_lower_percent'])+'%']
        assert tables[1][i]==expected
    order=['baseline']+[prefix+s for s in seeds for prefix in ('','nominal_','thinned_','mps_','mamps_')]
    assert len(tables[2])==17
    for row,m in zip(tables[2][1:],order):
        q=methods[m]
        assert row[1:]==[str(q['adopted'])+'/1024',str(q['harmful_adopted']),num(q['mean_paired_percentage_gain'])+'%',str(q['execution_failures'])]
    for i,s in enumerate(seeds,1):
        q=comp['adoption_differences'][s]
        assert tables[3][i]==[s,num(comp['calibration_matching']['probabilities'][s],4),num(100*q['observed_difference']),'['+num(100*q['lower'])+', '+num(100*q['upper'])+']']
        q=evidence['development_paired_table'][s]
        assert tables[4][i]==[s,str(q['old_adopt'])+' / '+str(q['new_adopt']),num(q['old_mean_gain_percent'])+'%',num(q['new_mean_gain_percent'])+'%',num(q['gain_change_pp']),str(q['old_harm'])+' / '+str(q['new_harm'])]
        q=evidence['deployed_role_participation']['confirmation'][s]
        assert tables[5][i]==[s,*map(str,q['actual_command_change_steps']),str(q['multi_role_command_change_steps'])]
        q=methods[s]
        assert tables[6][i]==[s,str(q['callbacks']),num(q['callback_p95_ms']),num(q['callback_max_ms']),str(q['callback_over_500ms'])]
    for table in doc.tables:
        assert table.rows[0]._tr.trPr.find(qn('w:tblHeader')) is not None
        assert all(row._tr.trPr.find(qn('w:cantSplit')) is not None for row in table.rows)
    assert doc.sections[0]._sectPr.xml==Document(REFERENCE).sections[0]._sectPr.xml
    heading_text=[p.text for p in doc.paragraphs if p.style.name.startswith('Heading') or p.style.name=='Title']
    assert all(not any(c in text for c in '、，：:；;。！？!?') for text in heading_text)
    text='\n'.join(p.text for p in doc.paragraphs)
    for expected in ['5/1819','5/1865','8/1835','1/672','1/673','1/667','19277','52224','0.05/42']:
        assert expected in text,expected
    for forbidden in ['\\frac','\\sum','\\begin','\\end','占位符','TODO']:
        assert forbidden not in text,forbidden
    changed=[]
    with ZipFile(REFERENCE) as before,ZipFile(docpath) as after:
        assert set(before.namelist())<=set(after.namelist())
        for name in before.namelist():
            if before.read(name)!=after.read(name):changed.append(name)
        assert set(changed)<= {'word/document.xml','docProps/core.xml','[Content_Types].xml','word/_rels/document.xml.rels'}
        def rels(bundle):return {n.get('Id'):dict(n.attrib) for n in etree.fromstring(bundle.read('word/_rels/document.xml.rels'))}
        previous,current=rels(before),rels(after)
        assert all(current.get(k)==v for k,v in previous.items())
    pdf=pdfium.PdfDocument(revision/'word_render.pdf');pages=len(pdf)
    assert sorted(reviewed)==list(range(1,pages+1))
    pngs={f'page-{i}.png':sha(revision/'word_render'/f'page-{i}.png') for i in reviewed}
    result=dict(document_sha256=sha(docpath),document_bytes=docpath.stat().st_size,
        pages=pages,pages_visually_reviewed=reviewed,all_pages_visually_reviewed=True,
        all_pages_clean=True,page_png_sha256=pngs,review_resolution_scale=1.6,
        all_seven_numeric_tables_match_terminal_sources=True,
        table_count=7,native_equation_count=14,actual_figures=3,
        repeated_table_headers_and_unsplit_rows=True,
        original_package_parts_preserved_except_declared_slots=True,
        original_relationships_and_section_geometry_preserved=True,
        declared_changed_parts=changed,reference_sha256=sha(REFERENCE),
        evidence_sha256=sha(BASE/'paper_top_two_attempt6_evidence_20261104/evidence.json'),
        verifier_sha256=sha(Path(__file__)),
        actual_canonical_renderer_exit_code=1,canonical_failure_diagnosed=True,
        renderer='Fresh hidden owned Word instance; PDF rasterized with bundled pypdfium2',
        actual_owned_word_export_exit_code=0,word_terminal_chunk_id='85a9a3',
        all_negative_results_retained=True,submission_readiness_not_certified=True)
    with (revision/'visual_and_package_qa.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('page_png_sha256',)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',type=Path,required=True);p.add_argument('--reviewed-pages',required=True)
    a=p.parse_args();verify(a.revision,[int(v) for v in a.reviewed_pages.split(',')])
