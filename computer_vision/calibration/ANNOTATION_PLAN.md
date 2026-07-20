# Cube, CS-200, and visual-QA annotation plan

This plan defines the next OBP-CV calibration work after the validated
checkerboard audit. The goal is a bundle-adjusted fixed-camera calibration in
named laboratory coordinates, not another camera-relative provisional graph.

## Recommended sequence

1. Confirm the physical dimensions and point naming for both targets.
2. Extract deterministic source frames with decoded frame index and presentation
   timestamp (PTS).
3. Annotate cube joints and CS-200 marker centers with visibility and uncertainty.
4. Run independent annotation review before fitting geometry.
5. Fit non-coplanar cube geometry, then align it to the CS-200 laboratory frame.
6. Produce epipolar and undistortion-grid overlays from held-out annotations.
7. Run robust joint bundle adjustment and publish train, held-out, known-length,
   closure, and uncertainty results.

Do not construct object coordinates until the source documentation confirms
whether the cube's stated 27.5 inches describes adjacent joint spacing or total
outer extent. Do not infer CS-200 marker IDs or dimensions from video pixels;
transcribe them from the manufacturer's technical drawing.

## Source and frame manifest

Create `annotations/source_frames.jsonl` with one record per extracted frame:

```json
{"schema_version":1,"system":"optitrack","camera":"15","video_filename":"opti_cube_cam15.mp4","video_sha256":"...","decoded_frame_index":0,"pts_seconds":0.0,"image_width":1280,"image_height":720,"frame_sha256":"..."}
```

- Preserve the video hashes already recorded by the calibration audit or add
  hashes for cube and ground-plane videos before annotation.
- Use shared frame indices only after confirming synchronization within a fixed
  camera system. Never align the iPhone by nominal frame index.
- Extract several sharp frames per camera. If a target moves, choose synchronized
  poses with broad image coverage; if it is static, use separated frames to
  measure repeatability rather than treating duplicates as independent geometry.

## Cube annotations

Store one record per camera/frame in `annotations/cube_points.jsonl`:

```json
{"schema_version":1,"frame_id":"optitrack_15:120","target":"cube_3x3x3","points":[{"point_id":"x0_y0_z0","u_px":412.3,"v_px":255.8,"visibility":"visible","uncertainty_px":0.5}]}
```

- Use stable topology IDs `x0_y0_z0` through `x2_y2_z2`; define axis directions
  with a photographed target orientation before annotation starts.
- Record only visible joint centers as measured points. Mark occluded, clipped,
  blurred, and ambiguous joints explicitly instead of estimating hidden pixels.
- Allow `uncertainty_px`, `annotator`, `reviewer`, and `notes` on every record.
- Annotate the same subset twice, blinded, to quantify inter-annotator error.
- Use the cube first for non-coplanar PnP and for connecting the currently
  disconnected Edgertronic checkerboard graph.

## CS-200 annotations

Store one record per camera/frame in `annotations/cs200_points.jsonl`:

```json
{"schema_version":1,"frame_id":"optitrack_15:240","target":"cs200","drawing_revision":"...","points":[{"point_id":"drawing_marker_id","u_px":631.1,"v_px":401.6,"visibility":"visible","uncertainty_px":0.4}]}
```

- Copy marker IDs, center coordinates, units, drawing revision, dimensional
  tolerances, origin, and axis definitions from the authoritative drawing.
- Annotate marker centers, not blob edges. When a marker is elliptical in the
  image, optionally retain fitted ellipse parameters as supporting evidence.
- Use CS-200 coordinates to define lab origin, vertical, and named horizontal
  axes only after checking handedness against the OBP coordinate convention.
- Propagate drawing tolerances and pixel uncertainty into the final alignment.

## Annotation QA

- Render labeled overlays for every annotated frame and require reviewer signoff.
- Reject points outside the image, duplicate IDs, impossible visibility states,
  and records whose source/frame hashes do not match the manifest.
- Report inter-annotator median, p95, and maximum pixel distance by target and
  camera. Resolve systematic disagreements before calibration.
- Hold out complete frames and camera-target observations, not random individual
  points from the same frame.
- Preserve rejected annotations and reasons in a separate audit file.

## Epipolar overlays

Generate `qa/epipolar/<pair>/<frame>.png` for every accepted OptiTrack edge and
for candidate Edgertronic edges after cube fitting.

- Undistort points before applying the fundamental or essential matrix.
- Draw the source point, target epipolar line, nearest point on that line, and
  signed point-to-line distance.
- Use held-out checkerboard, cube, and CS-200 observations with distinct colors.
- Publish per-pair median, p95, and maximum symmetric epipolar distance plus the
  number of held-out correspondences.
- Include at least one failure overlay for every rejected edge so thresholds are
  visually auditable rather than represented only by a CSV value.

## Undistortion grids

Generate `qa/undistortion/<camera>.png` and a machine-readable summary per camera.

- Show the source image boundary, principal point, observed calibration radius,
  full-frame radius, valid output ROI, and a regular grid before/after mapping.
- Overlay detected checkerboard rows and columns to expose residual curvature.
- Report radial scale and derivative minima, remap Jacobian sign, percentage of
  valid output pixels, and crop/alpha settings.
- Include both a representative real frame and a synthetic grid. The current
  monotonicity test remains a release-blocking invariant.

## Final calibration outputs

- `annotations/source_frames.jsonl`
- `annotations/cube_points.jsonl`
- `annotations/cs200_points.jsonl`
- `annotations/rejections.jsonl`
- `results/optitrack_lab_calibration.json`
- `results/edgertronic_lab_calibration.json`
- `results/bundle_adjustment_report.json`
- `qa/epipolar/` overlays and summary CSV
- `qa/undistortion/` overlays and summary CSV

The final report should include coordinate conventions, units, source hashes,
software versions, optimization loss, train and held-out reprojection error,
epipolar error, known-length error, loop closure, parameter uncertainty, and a
clear provisional or final status.
