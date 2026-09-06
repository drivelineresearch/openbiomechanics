# Swing viewer browser smoke

This check creates two small synthetic C3Ds in a temporary directory, serves the
viewer on a loopback ephemeral port, and exercises loading, overlay, scaling,
synchronization, camera controls, pre-load frame stepping, and missing data.
It checks behavior, not the scientific accuracy of marker-derived metrics.

From the repository root, in an isolated environment:

```bash
python3 -m pip install -r requirements-dev.txt playwright==1.62.0
python3 -m playwright install chromium
python3 tests/browser/swing_viewer_smoke.py
```

Linux CI uses `python3 -m playwright install --with-deps chromium` for system
libraries. `OBP_BROWSER_EXECUTABLE=/absolute/path/to/chromium` optionally selects
an existing Chromium for local checks. The default public unittest suite does
not require a browser; this smoke test has its own GitHub Actions job.
