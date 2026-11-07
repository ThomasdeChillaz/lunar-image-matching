"""Presentation-only lighting grid from already rendered and audited pilots."""
import json
from pathlib import Path

from north_terrain_pilot_v32 import pilot_plan,sha


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plan,_=pilot_plan();audit=json.loads(Path('reports/north_terrain_pilot_audit_v34.json').read_text())
    if not audit['passed']or audit['correct_oracle_poses']!=12:raise ValueError('Actual complete geometry audit required')
    root=Path('runs/north-v33-terrain-pilots');summary=json.loads((root/'summary.json').read_text())
    if audit['renderer_summary_sha256']!=sha(root/'summary.json'):raise ValueError('Audited render summary changed')
    output=Path('reports/north_pilot_lighting_v34.png')
    if output.exists():raise FileExistsError('Preserve earlier figure')
    fig,axes=plt.subplots(2,3,figsize=(11,8));sources={}
    for light,((elevation,azimuth),axis)in enumerate(zip(plan['rendered_sun_elevation_azimuth_degrees'],axes.flat)):
        path=root/'np31-pilot-0'/'view-0'/f'light-{light}'/'image.png';digest=sha(path)
        expected=next(r['image_sha256']for r in summary['records'][0]['images']if r['view']==0 and r['light']==light)
        if digest!=expected:raise ValueError('Rendered image changed')
        with Image.open(path)as im:axis.imshow(im.copy())
        axis.set_title(f'Sun elevation {elevation:g}°, azimuth {azimuth:g}°');axis.axis('off');sources[str(path)]=digest
    fig.suptitle('Northern terrain pilot: fixed camera, six rendered Sun directions',fontsize=13)
    fig.tight_layout();fig.savefig(output,dpi=140);plt.close(fig)
    Path('reports/north_pilot_figure_provenance_v34.json').write_text(json.dumps(dict(
        source_sha256=sha(__file__),input_images=sources,figure_sha256=sha(output),
        scope='Presentation only; original pixel intensities, no per-panel normalization. Constant diffuse simulation, not orbital photographs or matcher results.'),indent=2)+'\n')


if __name__=='__main__':render()
