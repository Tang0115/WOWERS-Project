"""Capture the four live-app screenshots used in Section 4.6 (Frontend
Visualization System): the National Opportunity Map (fig_m1_nationalmap.png),
a Plant Detail view (fig_m1_plantdetail.png), a per-state portfolio
(fig_m1_stateportfolio.png), and the Analytics view (fig_m1_analytics.png).

Unlike the other thesis figures, these are not derived from a parquet/geojson
— they are real screenshots of the running dashboard, captured headlessly so
the exact pixels are reproducible rather than hand-taken. Requires the Vite
dev server running at localhost:5173 and Playwright with the Chromium
browser installed (`pip install playwright && playwright install chromium`).

Run from the repository root, with the dev server already up in a second
terminal (`cd frontend && npm run dev`):
    python3 thesis/figures/make_fig_m1_screenshots.py
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
URL = "http://localhost:5173"
PLANT_ID = "OH0024732"  # Jackson Pike Water Resource Recovery Facility, Columbus OH
STATE_CODE = "CA"  # largest state portfolio: 147 viable of 156 scored


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 950}, device_scale_factor=2)

        page.goto(f"{URL}/", wait_until="networkidle")
        page.wait_for_timeout(2500)  # let MapLibre finish painting raster tiles
        page.screenshot(path=str(OUT / "fig_m1_nationalmap.png"))
        print("saved fig_m1_nationalmap.png")

        page.goto(f"{URL}/plant/{PLANT_ID}", wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "fig_m1_plantdetail.png"))
        print(f"saved fig_m1_plantdetail.png (plant {PLANT_ID})")

        # The remaining two views are Recharts-only (no MapLibre), but the charts
        # animate in on mount, so wait past the animation before capturing or the
        # bars are caught mid-transition at partial height.
        #
        # StatePortfolio gets its own wider viewport. At the 1600 px used above,
        # the portfolio table overflows its horizontal scroll container by 339 px
        # and the last columns are cut off mid-figure; 2100 px is the width at
        # which all ten columns fit with no clipping. A full-page capture was
        # tried and rejected: the two charts sit below a 50-row table (PAGE_SIZE
        # in StatePortfolio.tsx), which makes the image 2718 CSS px tall and the
        # table text about 3 pt once scaled onto a thesis page. The charts are
        # described in the prose instead.
        wide = browser.new_page(viewport={"width": 2100, "height": 1150}, device_scale_factor=2)
        wide.goto(f"{URL}/state/{STATE_CODE}", wait_until="networkidle")
        wide.wait_for_timeout(2000)
        wide.screenshot(path=str(OUT / "fig_m1_stateportfolio.png"))
        print(f"saved fig_m1_stateportfolio.png (state {STATE_CODE})")
        wide.close()

        page.goto(f"{URL}/analytics", wait_until="networkidle")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(OUT / "fig_m1_analytics.png"))
        print("saved fig_m1_analytics.png")

        browser.close()


if __name__ == "__main__":
    main()
