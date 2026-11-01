# Same-source pilot transport recovery

The original public four-pilot extraction stopped before returning any source
window: GDAL could not read the BigTIFF tile-offset/size array. Its protocol,
log, failed state and output folder are preserved. No renderer or training job
was launched from it. This is a transport failure, not a model outcome.

A full raw source download was then attempted under a600second cap. At its
observed rate the5,222,983,620byte file would exceed that allowance. It was
stopped after159,383,552bytes; that partial and the interrupted state remain
preserved. No partial file is called a complete source or decoded as terrain.

The replacement downloads the same header and exactly100encoded tiles covering
the original four pilot windows:53,976,152tile bytes plus131,072header bytes.
Every HTTP reply must have the exact206status, Content-Range, length and fixed
source ETag. Invalid replies are rejected before reading the body. Two I/O
workers,600seconds maximum, no other windows. The local sparse TIFF is explicitly
INCOMPLETE; only the declared pilot windows are permitted. Every acquired range
is hashed and verified before decoding. This is not a publisher full-file
checksum or a complete terrain download.

`acquire_north_pilots_v33.py` then reads these identical windows from the
verified cache and reuses the original v32meshing functions. The unchanged
public v32renderer uses a separate output directory. The v33auditor adds source
array/mesh integrity and independent reference-world/query-projection/in-frame
label replay; all original geometry, pose and oracle tolerances remain fixed.

```sh
python cache_north_pilot_tiles_v33.py
python acquire_north_pilots_v33.py
/path/to/pinned/Blender --background --factory-startup --threads 2 \
 --python-exit-code 1 --python render_north_pilots_v32.py -- \
 --data data/north-render-v31/pilots-v33-local \
 --output runs/north-v33-terrain-pilots
python verify_north_pilots_v33.py
```

Publish/hash-verify these sources before pixel decoding. The original1,800second
pilot render allowance remains unused and unchanged, with an external calendar
and monotonic deadline. Six transport tests pass, including bad/truncated range
replies, exact tile support and forbidden reserve access. Full regression passes278tests/four optional skips. No training, selection, calibration or final terrain is acquired here.
Passing these pilots cannot establish a learned gain, fresh released-LunarGeo
test or real orbital accuracy. All earlier failed attempts remain recorded.
