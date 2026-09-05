# Hitting swing viewer

Play back a hitting C3D in the browser as an animated 3D skeleton and reconstructed bat, or overlay two recorded swings for mechanical comparison.

![Dual swing comparison overlay](imgs/swing_overlay.gif)

## Features

- **Single swing playback & dual overlay**: Inspect an individual swing or overlay two recorded trials simultaneously.
- **Temporal synchronization**: Synchronize playback at estimated contact, swing start, peak bat speed, or recording percentage.
- **Spatial alignment**:
  - **Each other's hips (at sync)**: Aligns pelvis positions horizontally at the synchronized event while maintaining grounded feet.
  - **Each other's hips (stance)**: Aligns initial stance positions.
  - **Home plate**: Displays both batters in their recorded batter's box locations relative to plate.
- **Match player size**: Scales body segments and arm reach to normalize athlete stature so mechanics can be compared without height discrepancies.
- **Bat metrics & comparison**: Calculates peak barrel speed, contact barrel speed, attack angle, bat direction, time to contact, and exit velocity, displaying side-by-side values and differences ($\Delta$).
- **3D camera presets**: Quickly toggle between Pitcher, Behind plate, Left side, Right side, and Top-down views, or orbit freely by dragging.

![Dual swing comparison with metrics](imgs/overlay_comparison.png)

## Prerequisites

Install the repository dependencies and download the hitting C3D files:

```bash
scripts/download_data.sh --discipline hitting
```

## Run the viewer

From the repository root, run:

```bash
python3 swing_visualizer/app.py
```

The application opens `http://127.0.0.1:8765` in your default browser.

- Pass `--no-open` to start the server without automatically opening a browser window.
- Pass `--port PORT` to use a custom port.
- Pass `--data-root PATH` to load C3Ds from another directory.

![Single swing inspection](imgs/single_swing.png)

## Interpret the results

The viewer derives several values from marker trajectories. They are exploratory estimates, not recorded laboratory measurements:

- **Contact** is estimated as the frame with the highest reconstructed barrel speed.
- **Bat geometry** is fitted to the available rigid-body bat markers and rendered at standard proportions.
- **Ball path** is an exploratory illustration rendered with a fixed launch trajectory.
- **Exit velocity** is parsed from trial filenames ending in tenths of a mile per hour (e.g. `_954.c3d` = 95.4 mph).

