"""Playwright smoke test for the Streamlit GUI."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    """Open the Streamlit app and assert key UI text is present."""
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot = out_dir / "ui_smoke.png"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        # Streamlit keeps websockets open — never reaches networkidle.
        page.goto(
            "http://127.0.0.1:8501",
            wait_until="domcontentloaded",
            timeout=120_000,
        )
        page.get_by_text("happy-housing", exact=False).first.wait_for(timeout=90_000)
        page.wait_for_timeout(3000)
        body = page.locator("body").inner_text(timeout=60_000)
        assert "happy-housing" in body
        assert (
            "Offer board" in body
            or "Listing" in body
            or "zł/m" in body
            or "Model" in body
        )
        page.screenshot(path=str(screenshot), full_page=True)
        browser.close()
    print(f"UI smoke OK -> {screenshot}")


if __name__ == "__main__":
    main()
