"""Exercise the browser against synthetic C3Ds without downloading release data."""

import importlib.util
import os
import pathlib
import tempfile
import threading

import ezc3d
import numpy as np
from playwright.sync_api import sync_playwright

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "swing_app", REPO / "swing_visualizer/app.py"
)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def fixture(path, height):
    c = ezc3d.c3d()
    labels = ["Marker1", "Marker2", "Marker3", "LASI", "RASI", "LFHD", "LHEE", "RHEE"]
    c["parameters"]["POINT"]["LABELS"]["value"] = labels
    c["parameters"]["POINT"]["RATE"]["value"] = [360.0]
    c["parameters"]["POINT"]["UNITS"]["value"] = ["m"]
    points = np.zeros((4, len(labels), 40))
    points[:3, :, :] = np.array(
        [
            [0, 0, 0.9],
            [0.6, 0, 0.9],
            [0.65, 0.03, 0.9],
            [0, 0.1, 1],
            [0, -0.1, 1],
            [0, 0, height * 0.0254],
            [0, 0.1, 0.05],
            [0, -0.1, 0.05],
        ]
    ).T[:, :, None]
    points[0, :3, :] += 0.4 * np.sin(np.linspace(0, np.pi, 40))
    c["data"]["points"] = points
    c.write(str(path))


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        fixture(root / "000001_000001_70_180_R_001_900.c3d", 70)
        fixture(root / "000002_000002_75_190_R_001_950.c3d", 75)
        server = app.ApplicationServer(("127.0.0.1", 0), app.Handler)
        server.data_root = root
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as p:
                options = {"headless": True}
                if os.environ.get("OBP_BROWSER_EXECUTABLE"):
                    options["executable_path"] = os.environ["OBP_BROWSER_EXECUTABLE"]
                browser = p.chromium.launch(**options)
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}")
                page.wait_for_function(
                    "document.querySelector('#trialA').options.length === 2"
                )
                page.locator("#stepForward").click()
                page.locator("#stepBack").click()
                page.locator("#compare").check()
                page.locator("#load").click()
                page.wait_for_function("window.viewer.motions.length === 2")
                page.evaluate("window.viewer.pause()")
                assert "Contact is estimated" in page.locator("#metrics").inner_text()
                page.locator("#matchSize").check()
                assert (
                    abs(page.evaluate("window.viewer.motions[1].scale") - 70 / 75)
                    < 1e-9
                )
                for mode in ("hips", "hipsStance", "plate"):
                    page.select_option("#align", mode)
                for button in page.locator(".viewbar button").all():
                    button.click()
                for sync in ("swingStart", "peakSpeed", "percentage", "contact"):
                    page.select_option("#sync", sync)
                page.locator("#showBall").check()
                page.locator("#stepForward").click()
                assert page.evaluate("window.viewer.phase") > 0
                page.locator("#stepBack").click()
                assert not errors, errors
                server.data_root = root / "absent"
                page.reload()
                page.wait_for_function("document.querySelector('#load').disabled")
                assert "No hitting C3Ds found" in page.locator("#state").inner_text()
                page.locator("#stepForward").click()
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    print(
        "Swing viewer browser smoke passed (synthetic C3Ds, overlay, controls, empty data)."
    )


if __name__ == "__main__":
    main()
