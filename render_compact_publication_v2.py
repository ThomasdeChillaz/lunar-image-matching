"""Presentation-only polish around the frozen study analysis.

Measurements, operating points, curves and uncertainty calculations remain in
the hash-bound report_compact_v2.py. This wrapper changes labels, PDF metadata
and footer links; it does not modify that frozen source.
"""
import argparse
import json
from pathlib import Path

from dataset import sha256


def render(output='reports',outreach=None):
    protocol=json.loads(Path('reports/compact_freeze_v2.json').read_text())
    if sha256('report_compact_v2.py')!=protocol['comparison_assets']['report_compact_v2.py']:
        raise ValueError('Frozen analysis source differs')
    import matplotlib.axes
    import reportlab.platypus as flow
    original_ylabel=matplotlib.axes.Axes.set_ylabel
    original_annotate=matplotlib.axes.Axes.annotate
    original_paragraph=flow.Paragraph
    original_document=flow.SimpleDocTemplate

    def ylabel(axis,label,*args,**kwargs):
        if label=='Incorrect accepted outputs (%; nonlinear scale)':
            label='Incorrect accepted outputs (%)\nNonlinear scale'
        return original_ylabel(axis,label,*args,**kwargs)

    def annotate(axis,text,*args,**kwargs):
        offsets={'LoMa-B (504)':(-5,16),'LoMa-R (504)':(-5,-17),
                 'RoMa v2 (fast512)':(-5,5),'Original RoMa':(5,12),
                 'MatchAnything4x':(5,-16)}
        if text in offsets:
            kwargs['xytext']=offsets[text]
            if offsets[text][0]<0:kwargs['ha']='right'
        return original_annotate(axis,text,*args,**kwargs)

    def paragraph(text,*args,**kwargs):
        text=text.replace('contain36','contain 36').replace('equals28.5','equals 28.5')
        return original_paragraph(text,*args,**kwargs)

    class Document(original_document):
        def __init__(self,*args,**kwargs):
            kwargs.update(title='Compact lunar image matching: frozen south-pole study',
                          author='Thomas de Chillaz',
                          subject='Controlled registration, modern baselines, transfer and limitations')
            super().__init__(*args,**kwargs)

        def build(self,story,*args,**kwargs):
            def linked(original):
                def footer(canvas,doc):
                    original(canvas,doc)
                    label='github.com/ThomasdeChillaz/lunar-image-matching'
                    canvas.setFont('Helvetica',7)
                    canvas.drawRightString(553,35,label)
                    canvas.linkURL('https://'+label,(300,32,553,43),relative=0)
                return footer
            for name in ('onFirstPage','onLaterPages'):
                if name in kwargs:kwargs[name]=linked(kwargs[name])
            return super().build(story,*args,**kwargs)

    matplotlib.axes.Axes.set_ylabel=ylabel
    matplotlib.axes.Axes.annotate=annotate
    flow.Paragraph=paragraph
    flow.SimpleDocTemplate=Document
    try:
        from report_compact_v2 import build,outreach_text
        evidence=build(output=output)
        if outreach:
            Path(outreach).write_text(outreach_text(evidence))
        return evidence
    finally:
        matplotlib.axes.Axes.set_ylabel=original_ylabel
        matplotlib.axes.Axes.annotate=original_annotate
        flow.Paragraph=original_paragraph
        flow.SimpleDocTemplate=original_document


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports')
    parser.add_argument('--outreach',help='Save an unsent draft at this explicit local path')
    args=parser.parse_args()
    render(args.output,args.outreach)
