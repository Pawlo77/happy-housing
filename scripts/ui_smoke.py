"""Playwright smoke test for the multipage Streamlit GUI."""

from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    """Open Home and assert navigation + key copy."""
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot = out_dir / "ui_smoke.png"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(
            "http://127.0.0.1:8501",
            wait_until="domcontentloaded",
            timeout=120_000,
        )
        page.get_by_text("happy-housing", exact=False).first.wait_for(timeout=90_000)
        page.wait_for_timeout(2500)
        body = page.locator("body").inner_text(timeout=60_000)
        assert "happy-housing" in body
        assert "Home" in body or "Explore" in body or "Admin" in body
        assert (
            "Model" in body
            or "Data sources" in body
            or "methodology" in body.lower()
            or "Offers" in body
        )
        page.screenshot(path=str(screenshot), full_page=True)
        browser.close()
    print(f"UI smoke OK -> {screenshot}")


if __name__ == "__main__":
    main()
