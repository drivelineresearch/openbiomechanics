# Hitting swing viewer

Play back a hitting C3D in the browser as an animated skeleton and reconstructed bat, or overlay two swings for comparison.

## Prerequisites

Install the root Python requirements and download the hitting C3Ds:

```bash
scripts/download_data.sh --discipline hitting
```

## Run the viewer

From the repository root, run:

```bash
python3 swing_visualizer/app.py
```

The application opens `http://127.0.0.1:8765` in your default browser. Use `--no-open` to start the server without opening a browser. Use `--data-root PATH` to load C3Ds from another directory.

## Interpret the results

The viewer derives several values from marker trajectories. They are exploratory estimates, not recorded measurements:

- Contact is the frame with the highest reconstructed barrel speed.
- The bat is fitted to the available bat markers and rendered at 34 inches.
- The displayed ball path is an illustration with a fixed direction and launch angle.
- Confidence describes bat-marker coverage near estimated contact. It does not validate the estimated event or metrics.

The viewer reads an exit velocity from trial filenames that end in an underscore followed by tenths of a mile per hour, such as `_954.c3d`. It displays no recorded exit velocity when the filename does not match that convention.
