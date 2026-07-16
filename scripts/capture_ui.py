"""Screenshot every tab of the running app for the written report (report section 7.1).

The chart figures come from `export_report_assets.py`, which renders Plotly objects
directly and never opens a browser. These are the complement: the actual interface,
with its filter panel, its tabs and its AI answers, which only exists in a browser.

Run:
    1. streamlit run app.py --server.port 8502 --server.headless true
    2. python -m scripts.capture_ui
Output: report/figures/ui_*.png

LM Studio must be running: the AI Assistant and Agent shots wait for a real answer
rather than photographing an empty tab. That is the point - a screenshot of the
interface with no output in it demonstrates nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

URL = "http://localhost:8502"
OUT = Path("report/figures")

# 1440 wide: comfortably above the 1280px the brief asks the UI to support, and
# narrow enough that the sidebar and content still read at report scale.
VIEWPORT = {"width": 1440, "height": 1000}

# The model answers in seconds, not milliseconds. These waits are for a real
# answer to land, so they are generous on purpose.
ANSWER_TIMEOUT_MS = 180_000
SETTLE_MS = 2_500

# Breathing room left under the content when the blank page below it is cropped.
PAD_PX = 48


def _wait_idle(page) -> None:
    """Block until Streamlit has stopped rendering.

    `networkidle` is not enough and produced a blank first screenshot: the socket
    goes quiet while the script is still running, and the Overview tab's preset
    insights call the model, so a cold start renders seconds after the page loads.
    Streamlit publishes its own busy state - the status widget in the top right -
    so we wait for that to go away rather than guessing at a sleep.
    """
    try:
        page.wait_for_selector(
            '[data-testid="stStatusWidget"]', state="detached", timeout=ANSWER_TIMEOUT_MS
        )
    except PWTimeout:
        print("  WARN still busy after the timeout; capturing anyway")
    page.wait_for_timeout(SETTLE_MS)


def _trim_dead_space(path: Path) -> tuple[int, int]:
    """Crop the empty page below the content. Returns the (before, after) height.

    Every shot is a fixed 1440x1000 viewport, so a tab whose content ends halfway
    down carries half a page of blank white into the report - and the report has a
    20-page ceiling. Only uniform background rows are removed: the scan starts to
    the right of the sidebar (which is a solid grey column full height, and would
    defeat a whole-row test) and stops at the last row holding anything.
    """
    image = Image.open(path).convert("RGB")
    width, height = image.size
    content = image.crop((int(width * 0.30), 0, width, height))  # skip the sidebar
    grey = content.convert("L")
    # Anything darker than near-white counts as content.
    mask = grey.point(lambda level: 0 if level > 244 else 255, mode="1")
    box = mask.getbbox()
    if box is None:
        return height, height
    last_row = min(height, box[3] + PAD_PX)
    if last_row >= height:
        return height, height
    image.crop((0, 0, width, last_row)).save(path)
    return height, last_row


def _shot(page, name: str, full: bool = True) -> None:
    path = OUT / f"ui_{name}.png"
    page.screenshot(path=str(path), full_page=full)
    before, after = _trim_dead_space(path)
    size = path.stat().st_size
    trimmed = f"  trimmed {before}->{after}px" if after < before else ""
    flag = "  <-- suspiciously small, check it" if size < 60_000 else ""
    print(f"  OK   {path}  ({size:,} bytes){trimmed}{flag}")


def _open_tab(page, label: str) -> None:
    page.get_by_role("tab", name=label).click()
    _wait_idle(page)


def _try(step: str, fn) -> bool:
    """Run one capture step. A failure here must not cost us the later shots."""
    try:
        fn()
        return True
    except PWTimeout:
        print(f"  WARN {step}: timed out. Captured what was on screen.")
        return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # channel="chrome" drives the Chrome already installed on this machine
        # rather than downloading a second copy of Chromium.
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        try:
            page.goto(URL, wait_until="networkidle", timeout=60_000)
        except PWTimeout:
            print(f"ERROR: nothing is serving {URL}. Start the app first (see docstring).")
            return 1

        print("Capturing:")
        _wait_idle(page)  # a cold start renders the Overview insights via the model
        _shot(page, "overview")

        _open_tab(page, "Exploration")
        _shot(page, "exploration")

        # --- AI Assistant: ask a real question and wait for the real answer.
        _open_tab(page, "AI Assistant")
        page.get_by_role("button", name="Which sub-categories are losing us money?").click()
        answered = _try(
            "AI answer",
            lambda: page.get_by_text("How this was answered").wait_for(timeout=ANSWER_TIMEOUT_MS),
        )
        if not answered:
            print("       (is LM Studio loaded? the shot will show the error state)")
        page.wait_for_timeout(SETTLE_MS)
        _shot(page, "ai_assistant")

        # The reasoning panel is the evidence for B2, so it gets its own shot open.
        if answered:
            _try("trace panel", lambda: page.get_by_text("How this was answered").click())
            page.wait_for_timeout(SETTLE_MS)
            _shot(page, "ai_trace")

        _open_tab(page, "Anomalies")
        page.wait_for_timeout(SETTLE_MS * 2)  # Isolation Forest + narration
        _shot(page, "anomalies")

        # --- Agent: run the question that uses the external holiday API.
        _open_tab(page, "Agent")
        page.get_by_role("button", name="Do sales rise around US public holidays in 2016?").click()
        _try(
            "agent run",
            lambda: page.get_by_text("Reasoning chain").wait_for(timeout=ANSWER_TIMEOUT_MS),
        )
        page.wait_for_timeout(SETTLE_MS)
        _shot(page, "agent")

        browser.close()

    print(f"\nUI screenshots -> {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
