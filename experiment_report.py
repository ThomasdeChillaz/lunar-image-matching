"""Generate inspectable figures, a two-page research report, and an offline demo."""
import base64,html,json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def png_data(image):
    ok,buf=cv2.imencode('.png',image)
    if not ok:raise ValueError('PNG encoding failed')
    return 'data:image/png;base64,'+base64.b64encode(buf).decode()


def render_results(rows,summary,gallery,config,meta,output):
    methods=config['methods'];colors=['#64748b','#0d9488','#7c3aed'];threshold=config['primary_threshold_px']
    # Three independently useful figures: reference agreement, appearance strata, runtime.
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,kind,label in zip(axes,['real_image_synthetic_transform','real_repeat_observation'],['Known-transform controls','Repeat observations: map-reference agreement']):
        for i,method in enumerate(methods):
            values=[]
            for prep in config['preprocessing']:
                selected=[r for r in rows if r['dataset_kind']==kind and r['method']==method and r['preprocessing']==prep and r['split']=='test']
                values.append(sum(r['error_px'] is not None and r['error_px']<=threshold for r in selected)/len(selected) if selected else 0)
            ax.bar(np.arange(2)+(i-1)*.24,values,width=.24,label=method,color=colors[i])
        ax.set_xticks([0,1],['None','CLAHE']);ax.set_title(label,fontsize=10);ax.set_ylim(0,1.08);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel(f'Fraction within {threshold:g}px (all attempts)');axes[1].legend(fontsize=8)
    fig.suptitle('Held-out S3 region; map uncertainty is not quantified',fontsize=11);fig.tight_layout();fig.savefig(output/'agreement.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for i,method in enumerate(methods):
        selected=[r for r in rows if r['dataset_kind']=='real_repeat_observation' and r['method']==method and r['preprocessing']=='none']
        good=[r for r in selected if r['error_px'] is not None]
        ax.scatter([r['incidence_difference_deg'] for r in good],[r['error_px'] for r in good],label=method,color=colors[i],alpha=.8)
    ax.set(xlabel='Difference in incidence angle (degrees)',ylabel='Map-reference disagreement (px)',title='Repeat observations across all three regions; failed fits omitted here')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'illumination.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for i,method in enumerate(methods):
        values=[r['runtime_ms'] for r in rows if r['method']==method]
        ax.bar(i,float(np.median(values)),color=colors[i]);ax.text(i,float(np.median(values)),f'{np.median(values):.0f} ms',ha='center',va='bottom')
    ax.set_xticks(range(len(methods)),methods);ax.set(ylabel='Median warmed wall time (ms)',title='CPU timing, one repeat; model loading and I/O excluded');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output/'runtime.png',dpi=150);plt.close(fig)
    # Representative held-out examples chosen by pair ID, not by favorable outcome.
    held_out=[g for g in gallery if g['row']['split']=='test' and g['row']['method']==methods[-1]]
    examples=held_out[:1]+held_out[-1:] if len(held_out)>1 else held_out
    fig,axes=plt.subplots(max(1,len(examples)),3,figsize=(10,3.5*max(1,len(examples))),squeeze=False)
    for cells,example in zip(axes,examples):
        a,b=example['reference'],example['query'];matrix=example['matrix'];row=example['row']
        aligned=np.zeros_like(a) if matrix is None else cv2.warpAffine(b,cv2.invertAffineTransform(matrix),(a.shape[1],a.shape[0]))
        overlay=np.stack((a,aligned,a),axis=-1)
        for axis,im,title in zip(cells,[a,b,overlay],['Reference',row['pair_id']+' query','Warp overlay' if matrix is not None else 'No transform']):
            axis.imshow(im,cmap='gray',vmin=0,vmax=255);axis.set_title(title,fontsize=9);axis.axis('off')
    fig.suptitle('Held-out examples: first and last pair by ID; qualitative alignment only',fontsize=11)
    fig.subplots_adjust(hspace=.22,wspace=.06,top=.9,bottom=.02,left=.02,right=.98)
    fig.savefig(output/'gallery.png',dpi=140);plt.close(fig)
    table=make_table(rows,methods,threshold)
    report_md=markdown_report(rows,table,config,meta)
    (output/'REPORT.md').write_text(report_md)
    make_pdf(output,rows,table,config)
    demo=[]
    for example in gallery:
        a,b=example['reference'],example['query'];matrix=example['matrix']
        aligned=np.zeros_like(a) if matrix is None else cv2.warpAffine(b,cv2.invertAffineTransform(matrix),(512,512))
        demo.append(dict(row=example['row'],reference=png_data(a),query=png_data(b),aligned=png_data(aligned)))
    template=Path(__file__).with_name('demo_template.html').read_text()
    (output/'demo.html').write_text(template.replace('__DEMO_DATA__',json.dumps(demo)))


def make_table(rows,methods,threshold):
    table=[]
    for kind in ['real_image_synthetic_transform','real_repeat_observation']:
        for method in methods:
            for prep in ['none','clahe']:
                selected=[r for r in rows if r['split']=='test' and r['dataset_kind']==kind and r['method']==method and r['preprocessing']==prep]
                errors=[r['error_px'] for r in selected if r['error_px'] is not None]
                table.append(dict(kind=kind,method=method,preprocessing=prep,n=len(selected),registered=sum(r['estimated'] for r in selected),
                    within=sum(r['error_px'] is not None and r['error_px']<=threshold for r in selected),
                    median=float(np.median(errors)) if errors else None))
    return table


def markdown_report(rows,table,config,meta):
    text='''# Lunar image matching: reproducible pilot study

## Research question
How do classical and pretrained learned matchers behave when real lunar terrain is observed under different illumination, and how does that differ from performance on controlled warps?

## Data and frozen protocol
15 published LROC NAC map-projected crops from CNSFM, 5 per region; 12 anchored repeat-observation pairs. Entire regions are assigned S1 reserved, S2 development, S3 held-out. No fine-tuning, parameter search or test-driven selection was performed. See lunar_manifest.json for source IDs, acquisition times, angles, CRS, bounds and SHA256. Each pair is resampled independently to a shared 512px map grid with source nodata masks. All 12 pairs were visually inspected for terrain overlap; independent checkpoint accuracy is not established.

Three methods: NCC (translation), SIFT + partial-affine RANSAC, and pretrained DISK + LightGlue + the same estimator. None/CLAHE preprocessing is evaluated for every method. NCC rotation cases are explicitly out-of-model and separately identified in summaries. Feature budgets differ: SIFT is capped at 2000, DISK at 512, so this is a practical preset comparison, not a matched-feature-count experiment. Learned inference is warmed on procedural input and CPU timing excludes downloads/model loading. One repeat per condition limits timing precision.

Controlled tests use one actual source image per region and three known warps (translation, intensity change, rotation). They have exact geometry but do not simulate moving shadows. Repeat-observation metrics use the authors' map coordinates as an external reference. Map uncertainty is unknown; disagreement with that reference is not true registration/navigation error. No outcome should be claimed as reproduction of LIMA.

## Held-out S3 measurements

| Experiment | Method | Preprocessing | Pairs | Estimated | Within 3px | Median scored discrepancy (px) |
|---|---|---|---:|---:|---:|---:|
'''
    for r in table:
        value='n/a' if r['median'] is None else f'{r["median"]:.2f}'
        text+=f'| {"Exact controls" if r["kind"]=="real_image_synthetic_transform" else "Map reference"} | {r["method"]} | {r["preprocessing"]} | {r["n"]} | {r["registered"]} | {r["within"]} | {value} |\n'
    text+='''
## Findings and limits
Known-transform controls establish implementation behavior on actual lunar textures. Repeat observations expose major illumination/shadow changes and failures absent from simple intensity warps. A returned transform, a high inlier count, or agreement with the source map alone does not establish correct navigation. The current subset is a pilot: only one held-out geographic region and four held-out repeat pairs. Pairs share a reference image and are statistically dependent. Do not report pair-wise confidence intervals as though these were independent sites.

Source map accuracy is not provided, and a partial-affine model may not capture relief/viewpoint distortion. The checkpoint-review page allows independent, distributed point annotation; validated points with picking uncertainty can replace the map reference in the same evaluation pipeline. That scientific validation remains open. No separate DEM or physical rendering is used.

## Reproduction
Use Python 3.12; install requirements-lock.txt, then requirements-learned.txt without dependency upgrades. Run `python dataset.py fetch`, `python fetch_models.py`, and `python evaluate.py --output runs/reproduction`. Each run refuses overwrites and saves frozen inputs, source hashes, environment, per-pair JSON/CSV, split-specific summaries, three plots, a gallery, this report and the PDF. The committed results preserve the actual executed source and manifest hashes. Hardware-dependent timings may vary.

## References
Li et al., Remote Sensing 17(13), 2302 (2025), https://doi.org/10.3390/rs17132302; data https://github.com/Bin501/CNSFM . Lindenberger et al., LightGlue, ICCV 2023, https://arxiv.org/abs/2306.13643 . Tyszkiewicz et al., DISK, NeurIPS 2020, https://arxiv.org/abs/2006.13566 . Exact source revisions, licenses and weight hashes: THIRD_PARTY.md and model_lock.json.
'''
    return text


def make_pdf(output,rows,table,config):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
    from reportlab.lib.pagesizes import A4
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='SmallBody',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=7,textColor=colors.HexColor('#1f2937')))
    styles.add(ParagraphStyle(name='CaptionSmall',fontSize=8,leading=10,textColor=colors.HexColor('#475569'),spaceAfter=6))
    doc=SimpleDocTemplate(str(output/'REPORT.pdf'),pagesize=A4,rightMargin=38,leftMargin=38,topMargin=32,bottomMargin=32)
    story=[]
    def p(text,style='SmallBody'):story.append(Paragraph(text,styles[style]))
    p('Lunar image matching','Title');p('A reproducible pilot with classical and pretrained learned matching','CaptionSmall')
    p('<b>Question.</b> How does performance on controlled image warps compare with matching real repeat lunar observations under changing illumination?')
    p('<b>Data.</b> Fifteen LROC NAC map-projected crops from the published CNSFM release; 12 repeat pairs across Chang\'e 3, Apollo 11 and Peak Near Shackleton. Whole scenes form reserved (S1), development (S2) and held-out (S3) splits. No model training or parameter search was performed.')
    p('<b>Protocol.</b> Shared 512 x 512 map grid; original nodata masks; NCC, SIFT/RANSAC and pretrained DISK/LightGlue/RANSAC; none/CLAHE ablation. Exact controls apply three known warps to one real image per scene. Failures stay in all denominators.')
    p('<b>Reference limitation.</b> Real observations are scored against supplied map coordinates, whose accuracy is unknown. Those numbers measure map-reference disagreement, not independent registration or navigation accuracy. Exact controls do not reproduce moving lunar shadows.')
    story.append(Image(str(output/'agreement.png'),width=510,height=185))
    p('Held-out measurements','Heading2')
    data=[['Experiment','Method / preprocessing','N','Fits','<=3px','Median px']]
    for r in table:
        data.append(['Control' if r['kind']=='real_image_synthetic_transform' else 'Map ref.',r['method'].replace('disk_lightglue','DISK+LG')+' / '+r['preprocessing'],str(r['n']),str(r['registered']),str(r['within']),'-' if r['median'] is None else f'{r["median"]:.2f}'])
    t=Table(data,colWidths=[65,190,30,35,45,65]);t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2e8f0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#94a3b8'))]));story.append(t)
    p('Median is over scorable estimates only; the <=3px column counts every attempt. Control rotation is outside NCC\'s translation model. All thresholds and failure details are in results.json.','CaptionSmall')
    story.append(PageBreak());p('Failure analysis and reproducibility','Title')
    story.append(Image(str(output/'gallery.png'),width=500,height=350))
    p('The first and last held-out pairs by ID are shown, with DISK/LightGlue overlays. This fixed selection is qualitative; an overlay is not ground truth. Green/magenta separation shows residual disagreement.','CaptionSmall')
    p('<b>What this study establishes.</b> The complete pipeline runs on real lunar observations and on exact-transform controls. It records estimated transforms, failures, reference disagreement, mask support and CPU timing. Pretrained weights and source/data versions are pinned and checksummed.')
    p('<b>Limits.</b> Only one held-out scene and four repeat pairs; shared reference images make observations dependent. SIFT uses up to 2000 features and DISK 512. Map accuracy, solar azimuth and independent checkpoint uncertainty are unavailable. A partial-affine transform may not explain terrain relief and viewpoint changes. Timing uses one warmed repeat per condition and is descriptive.')
    p('<b>Next scientific check.</b> The included offline landmark-review tool supports at least ten distributed, independently annotated point pairs with picking uncertainty. Import these points and rerun the frozen evaluation before claiming navigation accuracy. No separate terrain model or LIMA reproduction is claimed.')
    p('<b>Reproduce.</b> Install the pinned environment; run dataset.py fetch, fetch_models.py, then evaluate.py --output runs/reproduction. Frozen configurations, asset/source hashes, per-pair results and environment metadata accompany every run.')
    p('Sources: Li et al., Remote Sensing 17(13), 2302 (2025), doi:10.3390/rs17132302; CNSFM data release. Lindenberger et al., LightGlue (ICCV 2023), arXiv:2306.13643. Tyszkiewicz et al., DISK (NeurIPS 2020), arXiv:2006.13566. See THIRD_PARTY.md for notices.','CaptionSmall')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#64748b'));canvas.drawString(38,18,'Independent portfolio research | No NASA/JPL affiliation or endorsement');canvas.drawRightString(A4[0]-38,18,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
