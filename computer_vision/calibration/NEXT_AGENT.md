# Handoff: color targets, cube/ground plane, and final calibration

The checkerboard audit has produced provisional intrinsics for 13 feeds, full
pair transforms for attempted synchronized fits, and a connected provisional
OptiTrack pose graph in camera-19 coordinates. Continue from the committed
files in this directory; do not silently replace them with hand-entered values.

## Highest-priority geometry work

1. Annotate or deterministically detect the 3 × 3 × 3 cube joint grid in every
   OptiTrack and Edgertronic cube video. Confirm from the source documentation
   whether 27.5 inches is the spacing between adjacent red joint centers or the
   total outer dimension before constructing object coordinates.
2. Annotate the CS-200 marker centers using its published technical drawing.
   Record the exact marker-center coordinate convention and uncertainties.
3. Use the cube for robust non-coplanar PnP and multicamera correspondence,
   especially for the disconnected Edgertronic checkerboard graph.
4. Use the CS-200 to define the lab origin, vertical direction, and named axes.
   Do not conflate camera-19 coordinates with lab coordinates.
5. Resolve checkerboard 180° orientation using the blue orientation marks for
   OptiTrack. For monochrome feeds, use temporal continuity, cube geometry, or
   explicit correspondence labels.
6. Run a joint robust bundle adjustment over accepted checkerboard, cube, and
   ground-plane observations. Report train and held-out reprojection error,
   loop closure, triangulated known-length error, and parameter uncertainty.
7. Keep rejected pair fits as evidence; never force every pair into the graph.

## Color and grayscale work

1. Locate and rectify the SpyderCHECKR 24 target per camera/frame. Store source
   frame index/PTS, quadrilateral corners, orientation, and patch masks.
2. Use eroded patch interiors and robust medians. Record projected patch width,
   clipping, blur, compression variance, and within-patch MAD so tiny patches
   can be excluded deterministically.
3. Separate task levels:
   - grayscale neutrality, exposure, and monotone tone response;
   - inter-camera appearance matching;
   - regularized 3 × 3 or HSL correction with held-out patches;
   - absolute colorimetry only if a measured target reference and linear/RAW
     capture become available.
4. Do not apply RGB correction to Edgertronic monochrome footage. Restrict it
   to exposure, black/white level, gamma/tone, and spatial-uniformity tasks.
5. Use the neutral ramp to test monotonicity and channel balance. Do not infer a
   sensor spectral response from H.264/MOV pixel values.
6. Fit on a subset of patches/frames and score held-out patches/frames. Compare
   against identity, per-channel gain, and simple white-balance baselines so a
   more complex correction must demonstrate real generalization.
7. If reporting ΔE, state the color space, illuminant, observer, transfer
   function, target reference source, and whether pixels were linearized. Do
   not publish absolute ΔE from guessed internet patch values.

## Timing and validation rules

- The fixed OptiTrack feeds are the appropriate synchronized multicamera set.
- Do not assume the iPhone is genlocked. Its median interval is approximately
  4.167 ms, while the 95th-percentile interval is 12.5 ms and longer intervals
  occur. Use frame presentation timestamps.
- Treat rolling-shutter motion as a separate model/uncertainty term.
- Preserve byte-level source provenance: video filename, file ID or Drive URL,
  resolution, PTS, decoded frame index, and calibration-object dimensions.
- Every released matrix needs a coordinate convention, units, frame name,
  validation split, error statistics, and a clear provisional/final status.

## Expected next deliverables

- `annotations/cube_points.jsonl`
- `annotations/cs200_points.jsonl`
- `annotations/spydercheckr_quads.jsonl`
- a final bundle-adjusted fixed-camera calibration with lab-frame extrinsics;
- an Edgertronic rig derived from cube/ground-plane correspondences;
- a color/grayscale feasibility table with minimum patch-resolution rules;
- deterministic held-out QA reports and visual overlays.
