# Scene 1 stereo geometry executable

The stereo audit is prepared for a separate frozen execution. Its six generated controls cover subpixel registration, axial metres, ray centimetres, two row orders, an ambiguous row order, a reversed baseline, a vertical misalignment, invalid terrain and depth support, and an unrecognized palette. The controls do not open the archived samples.

The audit samples textured image patches and measures left-to-right and right-to-left native correspondence before opening the PFM sample values. It rejects low normalized correlation, a close competing peak and an inconsistent reverse track. The terrain-mask mode uses only the publisher's exact left-image terrain palette. It excludes sky and unrecognized or transparent pixels. An unrecognized opaque palette fails the affected case and blocks the aggregate conclusion. Without a mask, any depth consensus is explicitly a diagnostic. [Publisher palette](https://github.com/zqyu9/LuSNAR-dataset/blob/9c4a6073722be7268e65830c796aff67e4dd1576/README.md#-semantic-categories)

Each hypothesis is scored on the same native tracks where both row interpretations have positive, locally stable depth. The audit checks the baseline sign and vertical reprojection residual before comparing axial and ray depth, metre/centimetre/millimetre units, and top/bottom row order. The published 610.17784-pixel focal length and 0.31-metre baseline stay fixed. Both plausible principal points are reported and neither is declared verified. The saved track arrays allow independent arithmetic replay. A model requires adequate spatial coverage, low error and a margin over the runner-up. Aggregate masked consensus requires three or more decisive cases and no conflict among them. [Rig specification](https://arxiv.org/html/2407.06512v3#S3.SS3)

Once the source, runtime and receipts are frozen for the new admission, use the following command from this directory under the external calendar guard.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python \
  audit_lusnar_scene1_stereo_v117.py \
  --freeze extraction-freeze.json --extracted extracted \
  --receipt-sha256 c2911a8fc41f5ca662cb674ff5ef0ae118cc9f320fa1e24c0a572d8d8e805191 \
  --semantic-root temporal_and_semantic/stereo_semantic \
  --semantic-receipt temporal_and_semantic/stereo_semantic/COMPLETE.json \
  --semantic-receipt-sha256 5243bf02e92640d4d8c3df5b23500b7a1dc2e542e10d56261fa0572c472c57dd \
  --output scene1-stereo-audit.json
```

The stereo gate evaluates the released depth under the documented rig. It does not verify the rover pose, lever arm, principal point, absolute map frame or performance on scenes 7–9.
