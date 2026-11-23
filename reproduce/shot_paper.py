"""Screenshots of the app for Figure fig:gui, already cropped.

Usage: python shot_paper.py [URL]   (default http://localhost:8501/)
Writes app_home.png, app_setup.png, app_report.png and app_section.png next to
this script; make_figures.py assembles them into images/fig_gui.png.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"
HERE = os.path.dirname(os.path.abspath(__file__))


def idle(page, t=1.5):
    time.sleep(0.8)
    for _ in range(400):
        if page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        time.sleep(0.5)
    time.sleep(t)


def column(page):
    b = page.locator("[data-testid='stMainBlockContainer']").first.bounding_box()
    pad = 16
    return b["x"] + pad * 3, b["width"] - pad * 6


def shot(page, name, y0, y1):
    x, w = column(page)
    page.screenshot(path=os.path.join(HERE, f"app_{name}.png"), clip={"x": x, "y": y0, "width": w, "height": y1 - y0})
    print("wrote", name, int(y1 - y0))


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    page.goto(URL, wait_until="networkidle")
    idle(page, 4)
    tiles = page.locator(".st-key-tiles").first.bounding_box()
    shot(page, "home", 70, tiles["y"] + tiles["height"] + 10)

    page.get_by_role("button", name="Try it with demo data").click()
    page.wait_for_selector("text=Design wave height", timeout=300000)
    page.set_viewport_size({"width": 1440, "height": 6000})
    idle(page, 3)
    head = page.get_by_text("Design wave height", exact=True).first.bounding_box()
    lead = page.locator(".lead").first.bounding_box()
    shot(page, "report", head["y"] - 12, lead["y"] + lead["height"] + 12)
    s0 = page.get_by_text("Change through time", exact=True).first.bounding_box()
    nxt = [page.get_by_text(t_, exact=True) for t_ in ("Where the storms come from", "Trends in the annual means")]
    s1 = min((n.first.bounding_box() for n in nxt if n.count()), key=lambda b_: b_["y"])
    shot(page, "section", s0["y"] - 14, s1["y"] - 40)

    page.set_viewport_size({"width": 1440, "height": 900})
    page.locator(".st-key-topbar [data-testid='stButtonGroup']").first.locator("button[role='radio']", has_text="Setup").first.click()
    idle(page, 2)
    page.locator("[data-testid='stButtonGroup']").filter(has_text="Compare all").locator("button", has_text="Compare all").click()
    idle(page, 1)
    page.get_by_test_id("stCheckbox").get_by_text("Advanced options").click()
    idle(page, 1)
    page.set_viewport_size({"width": 1440, "height": 3000})
    idle(page, 2)
    top = page.get_by_text("1 · Data").first.bounding_box()
    run = page.get_by_role("button", name="Run analysis").bounding_box()
    shot(page, "setup", top["y"] - 14, run["y"] + run["height"] + 12)
    b.close()
