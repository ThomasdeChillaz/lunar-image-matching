"""Generate a local, offline landmark annotation page for independently checked points."""
import argparse,base64,json
from pathlib import Path
import cv2
from dataset import validate_manifest,load_pair


def make_review(manifest,root,output):
    assets=validate_manifest(manifest,root); pairs=[]
    for p in manifest['pairs']:
        a,b,_,_,_=load_pair(p,assets,root)
        images=[]
        for im in (a,b):
            ok,buf=cv2.imencode('.png',im)
            if not ok: raise ValueError('Encoding failed')
            images.append('data:image/png;base64,'+base64.b64encode(buf).decode())
        pairs.append(dict(pair_id=p['pair_id'],images=images))
    template=Path(__file__).with_name('checkpoint_template.html').read_text()
    Path(output).write_text(template.replace('__PAIR_DATA__',json.dumps(pairs)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='lunar_manifest.json');p.add_argument('--output',default='runs/checkpoint_review.html');a=p.parse_args()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    make_review(json.loads(Path(a.manifest).read_text()),'.',a.output)
    print(a.output)
