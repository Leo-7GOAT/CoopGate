"""Compose the full manuscript only after every original mainline stage ends.

No live outcomes, new inference, simulation, model selection or overwrite.
Render and inspect every final page before copying this document to Desktop.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from summarize_completed_ood_20261027 import read_complete as completed_ood

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts/paper_supplement_20261026/CoopGate_论文实验与理论补充_20261026.docx'
REFERENCE_SHA = 'd4310c3a98cae08438cb46db8f8414710c41b2f4d57c39f82fc386b170729ec6'
OUT = ROOT / 'runs/manuscript_completed_20261101/revision1'
NAME = 'CoopGate_论文完整初稿_20261101.docx'
JOINT = ROOT / 'artifacts/joint_risk_recovery_20261031/analysis'
FAIR = ROOT / 'artifacts/v44/fair_failure_preserving_execution'
FACTORIAL = ROOT / 'artifacts/candidate_factorial/complete_fair_cases'
CSPI = ROOT / 'artifacts/literature_reproduction/cspi_factorial_prospective'
INPUTS = {}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    INPUTS[Path(path).relative_to(ROOT).as_posix()] = digest(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def evidence():
    # No result-file access precedes the actual completed queue gate.
    events = [json.loads(line) for line in
              (ROOT / '.git/mainline-fair-replay-20261101.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    if not any(e['kind'] == 'queue_completed' for e in events):
        raise ValueError('The original factorial and CSPI queue is still incomplete; no outcomes read')
    for script, command in [('tools/execute_fair_failure_v44.py', 'summarize'),
                            ('tools/finalize_episode_keys.py', 'factorial'),
                            ('tools/finalize_episode_keys.py', 'cspi-export'),
                            ('tools/finalize_episode_keys.py', 'cspi-analysis')]:
        exits = [e for e in events if e['kind'] == 'command_exit'
                 and e['detail'].get('arguments', [])[1:3] == ['utf8', script]
                 and len(e['detail']['arguments']) >= 4
                 and e['detail']['arguments'][3] == command]
        if len(exits) != 1 or exits[0]['detail']['exit_code'] != 0:
            raise ValueError('A complete actual zero exit is required: ' + command)
    joint = read(JOINT / 'completion_receipt_20261031.json')
    if (joint['native_finalizer_exit_code'] != 0 or joint['actual_unique_runs'] != 16384
            or not joint['full_scheduled_key_coverage_exact']
            or digest(JOINT / 'certificate.json') != joint['certificate_sha256']):
        raise ValueError('Complete original joint study required')
    read(JOINT / 'certificate.json')
    fair_receipt = read(ROOT / 'artifacts/fair_replay_recovery_20261101/completion.json')
    if fair_receipt['native_finalizer_exit_code'] != 0 or fair_receipt['original_fair_sumo_runs'] != 5120:
        raise ValueError('Complete fair review required')
    for name in ('report', 'replay'):
        if digest(FAIR / f'analysis/{name}.json') != fair_receipt[f'{name}_sha256']:
            raise ValueError('Fair finalization changed')
    fair = read(FAIR / 'analysis/report.json')
    if read(FACTORIAL / 'execution/exit.json')['exit_code'] != 0:
        raise ValueError('Full factorial collection must end zero')
    verification = read(FACTORIAL / 'analysis/verification.json')
    if verification['actual_sumo_runs'] != 3200:
        raise ValueError('All factorial episodes required')
    for name in ('report', 'replay'):
        if digest(FACTORIAL / f'analysis/{name}.json') != verification[f'{name}_sha256']:
            raise ValueError('Full factorial review changed')
    factorial = read(FACTORIAL / 'analysis/report.json')
    cspi = read(CSPI / 'analysis/complete.json')
    if cspi['evaluated_independent_parent_clusters'] != 128 or cspi['new_sumo_runs'] != 0:
        raise ValueError('Original held-out CSPI scope required')
    for name, expected in cspi['files_sha256'].items():
        if digest(CSPI / 'analysis' / name) != expected:
            raise ValueError('Finalized author-code analysis changed')
    ood = completed_ood()
    ood_path = ROOT / 'artifacts/recovery_20261026/ood_recovery/analysis/report.json'
    INPUTS[ood_path.relative_to(ROOT).as_posix()] = digest(ood_path)
    failures = read(ROOT / 'artifacts/recovery_20261027/ood_failure_types/summary.json')
    return joint, fair_receipt, fair, factorial, cspi, ood, failures


def run_math(text):
    run = OxmlElement('m:r')
    value = OxmlElement('m:t')
    value.text = text
    run.append(value)
    return run


def scripted(base, sub=None, sup=None):
    tag = 'm:sSubSup' if sub is not None and sup is not None else ('m:sSub' if sub is not None else 'm:sSup')
    element = OxmlElement(tag)
    for name, text in [('m:e', base), ('m:sub', sub), ('m:sup', sup)]:
        if text is not None:
            child = OxmlElement(name)
            child.append(run_math(text))
            element.append(child)
    return element


def fraction(numerator, denominator):
    element = OxmlElement('m:f')
    for name, text in [('m:num', numerator), ('m:den', denominator)]:
        child = OxmlElement(name)
        child.append(run_math(text))
        element.append(child)
    return element


def build():
    if OUT.exists():
        raise FileExistsError('Retain every manuscript revision')
    if digest(REFERENCE) != REFERENCE_SHA:
        raise ValueError('Retained style reference changed')
    joint, fair_receipt, fair, factorial, cspi, ood, failures = evidence()
    doc = Document(REFERENCE)
    section = doc.sections[0]._sectPr.xml
    for child in list(doc.element.body):
        if child.tag != qn('w:sectPr'):
            doc.element.body.remove(child)
    doc.core_properties.title = '协同学习控制的基线相对门控与扰动性能风险验证'
    doc.core_properties.subject = 'CoopGate 完整研究论文'
    doc.core_properties.author = '李东旭'
    doc.core_properties.version = '20261101'

    def p(text, style=None):
        paragraph = doc.add_paragraph(text, style)
        paragraph.paragraph_format.first_line_indent = Cm(0 if style else .74)
        if not style:
            paragraph.paragraph_format.space_after = Pt(5)
        return paragraph

    def h(text, level=1):
        return p(text, f'Heading {level}')

    def eq(*nodes):
        paragraph = p('')
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.keep_together = True
        math = OxmlElement('m:oMath')
        for node in nodes:
            math.append(run_math(node) if isinstance(node, str) else node)
        paragraph._p.append(math)

    def table(caption, headers, rows, widths):
        caption_p = p(caption, 'Caption')
        caption_p.paragraph_format.keep_with_next = True
        t = doc.add_table(rows=1, cols=len(headers))
        t.autofit = False
        for column, width in zip(t.columns, widths):
            column.width = Cm(width)
        for cell, value in zip(t.rows[0].cells, headers):
            cell.text = value
        for row in rows:
            for cell, value in zip(t.add_row().cells, row):
                cell.text = str(value)
        borders = OxmlElement('w:tblBorders')
        for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            node = OxmlElement('w:' + edge)
            for key, value in {'val': 'single', 'sz': '4', 'color': 'D9D9D9'}.items():
                node.set(qn('w:' + key), value)
            borders.append(node)
        t._tbl.tblPr.append(borders)
        for ri, row in enumerate(t.rows):
            props = row._tr.get_or_add_trPr()
            props.append(OxmlElement('w:cantSplit'))
            if ri == 0:
                props.append(OxmlElement('w:tblHeader'))
            for ci, (cell, width) in enumerate(zip(row.cells, widths)):
                cell.width = Cm(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tc = cell._tc.get_or_add_tcPr()
                margins = OxmlElement('w:tcMar')
                for edge in ('top', 'left', 'bottom', 'right'):
                    n = OxmlElement('w:' + edge)
                    n.set(qn('w:w'), '80')
                    n.set(qn('w:type'), 'dxa')
                    margins.append(n)
                tc.append(margins)
                if ri == 0:
                    fill = OxmlElement('w:shd')
                    fill.set(qn('w:fill'), 'DCE6F1')
                    tc.append(fill)
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    paragraph.paragraph_format.first_line_indent = Cm(0)
                    paragraph.paragraph_format.space_after = Pt(3)
                    paragraph.paragraph_format.space_before = Pt(3)
                    paragraph.paragraph_format.line_spacing = 1.05
                    for run in paragraph.runs:
                        run.font.size = Pt(10)
                        run.bold = ri == 0
        p('')

    def number(value, places=3):
        return '不可评估' if value is None else f'{value:.{places}f}'

    # Full scientific text is kept separate from the guarded presentation code.
    from completed_manuscript_text_20261101 import compose
    compose(p, h, eq, table, number, joint, fair_receipt, fair, factorial, cspi, ood, failures)
    if doc.sections[0]._sectPr.xml != section:
        raise ValueError('Reference page geometry changed')
    for name, expected in INPUTS.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Completed input changed during authoring: ' + name)
    OUT.mkdir(parents=True, exist_ok=False)
    target = OUT / NAME
    doc.save(target)
    with ZipFile(REFERENCE) as original, ZipFile(target) as final:
        allowed = {'word/document.xml', 'docProps/core.xml'}
        if set(original.namelist()) != set(final.namelist()):
            raise ValueError('Unexpected template package membership change')
        for name in original.namelist():
            if name not in allowed and original.read(name) != final.read(name):
                raise ValueError('Preserve-only template component changed: ' + name)
    if digest(REFERENCE) != REFERENCE_SHA:
        raise ValueError('Original reference modified')
    record = {'utc': datetime.now(timezone.utc).isoformat(),
              'document_sha256': digest(target), 'reference_sha256': REFERENCE_SHA,
              'builder_sha256': digest(__file__),
              'text_source_sha256': digest(ROOT / 'operations/completed_manuscript_text_20261101.py'),
              'completed_inputs_sha256': INPUTS,
              'native_equations': len(list(doc.element.body.iter(qn('m:oMath')))),
              'new_inference': False, 'live_partial_outcomes_read': False,
              'template_opaque_parts_byte_preserved': True,
              'final_visual_review_required': True}
    (OUT / 'build.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(target), 'native_equations': record['native_equations']}, ensure_ascii=False))


if __name__ == '__main__':
    build()
