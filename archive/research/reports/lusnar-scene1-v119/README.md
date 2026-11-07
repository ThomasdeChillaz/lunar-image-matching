# LuSNAR Scene 1 camera and depth convention preparation

The exposed development scene supplies 1,094 eligible left image, depth and
semantic-mask timestamps. The metadata-only schedule selects 30 temporal
pairs in five time windows. The 60 endpoints are distinct from one another
and from the five stereo anchors. Each window retains two logged translation
bands and low, middle and high rotation choices. The greatest within-window
rotation ranges from 0.53° to 6.43°. The five windows share one simulated
terrain, and the lowest-rotation window has little leverage on camera origin.
[Source rig and dataset](https://arxiv.org/html/2407.06512v3#S3.SS3)

## Prepared assets

- [Stereo triplets](stereo-extraction.json) pin 18 verified members and
  [semantic masks](stereo-semantic-extraction.json) pin five additional
  stereo anchor labels.
- [Temporal pairs](temporal-extraction-freeze.json) pin the ordered
  frame identities, two motion bands and three selection versus two
  confirmation windows. [Temporal extraction](temporal-extraction.json)
  pins 180 image, depth and label files, totaling 353,967,285 bytes.
- The freeze reproduced byte for byte. Each extracted temporal and semantic
  payload independently matched its source CRC and saved SHA-256. Header
  checks confirmed PNG and PFM structure. The audit has yet to decode those
  image and depth samples.

The ZIP is an incomplete local Scene 1 prefix. The exact inventory and
original CRC-verified metadata must accompany it. To reproduce extraction,
place the two preparation scripts together in a working directory, with
`lusnar-metadata` and `lusnar_metadata_audit_v46.py` in its parent directory.
Run the preparation with fresh output names. The original five-triplet
freeze remains separate from the additional temporal and semantic evidence.

## Validation sequence

1. Under a separate bounded dispatch, measure reciprocal stereo disparity
   before reading depth. Score each row order, axial/ray interpretation
   and physical unit on the same labelled terrain support. Require at least
   three decisive stereo pairs with no conflicts or unknown palette.
   [Stereo protocol and command](STEREO_AUDIT.md)
2. Write and seal image-only temporal tracks. The scoring stage then joins
   the selected stereo depth convention, the logged poses and the independent
   later depth. Compare 24 fixed hypotheses for quaternion direction, the
   origin of the logged translation, camera pitch sign and principal point.
   Keep all 30 pairs in the report. Require enough observations in every
   time window and report ambiguity if selection and confirmation do not
   share exactly one qualifying convention.
3. Use a separate exact geometry control after confirmation. Reproject
   reference surface points and require later depth support. Convert native
   1024-pixel centres to the 512-pixel comparison coordinates, then run
   the original 3-pixel AP3P estimator. Require at least 25 eligible pairs,
   at least four per window and 95% correct poses among eligible pairs.

The procedural tests cover disparity and occlusion, depth formats and
units, unknown palette, the optical coordinate frame, pixel-centre
conversion, image-only optical tracking, ambiguity and the pose oracle.
Actual stereo, temporal and pose checks await execution under a
source-published dispatch after the ongoing benchmark. Scene 1 can
qualify a geometry pipeline. It provides no held-out transfer result
for scenes 7–9, and no reserved scene was accessed.

[Original format and gate](GEOMETRY_GATE.md) ·
[LuSNAR project](https://luoxubo.github.io/LuSNAR-dataset/)
