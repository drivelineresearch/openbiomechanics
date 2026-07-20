Legacy single-video 2-D calibration tutorial

Run frameprocessing.py first to extract candidate frames, then run
2dcalibrate.py to process that tutorial's calibration sequence.

This directory is not the current OBP-CV multi-camera calibration pipeline and
its sample output must not be substituted for the validated provisional camera
models. For current intrinsics, held-out relative extrinsics, source hashes,
timing diagnostics, and reproduction instructions, use:

    ../calibration/README.md

For the planned cube and CS-200 annotations, epipolar overlays, undistortion
grids, and final lab-frame bundle adjustment, use:

    ../calibration/ANNOTATION_PLAN.md
