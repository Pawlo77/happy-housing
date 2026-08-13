"""Capture README screenshots from a running Streamlit app.

Requires ``make app`` (or equivalent) on http://127.0.0.1:8501.
Writes PNGs under ``docs/screenshots/``.
"""

from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path("docs/screenshots")
BASE = "http://127.0.0.1:8501"
VIEWPORT = {"width": 1440, "height": 1000}


def _shot(page: Page, name: str) -> None:
    path = OUT / name
    page.screenshot(path=str(path), full_page=True)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def _nav(page: Page, label: str) -> None:
    page.get_by_role("link", name=label).first.click()
    page.wait_for_timeout(3500)


def _tab(page: Page, label: str) -> None:
    tab = page.get_by_role("tab", name=label).first
    tab.scroll_into_view_if_needed()
    try:
        tab.click(timeout=15_000)
    except Exception:
        tab.click(force=True, timeout=15_000)
    page.wait_for_timeout(2500)


def main() -> None:
    """Capture Home / Explore / Stats / Admin screenshots for the README."""
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(BASE, wait_until="domcontentloaded", timeout=120_000)
        page.get_by_role("link", name="Home").first.wait_for(timeout=60_000)
        page.wait_for_timeout(3000)

        _shot(page, "01-home.png")
        _tab(page, "Market pulse")
        _shot(page, "01b-home-pulse.png")
        _tab(page, "Model")
        _shot(page, "02-home-model.png")
        _tab(page, "Data sources")
        _shot(page, "03-home-sources.png")

        _nav(page, "Explore")
        page.wait_for_timeout(4000)
        _shot(page, "04-explore-map.png")
        _tab(page, "Board")
        _shot(page, "05-explore-board.png")
        _tab(page, "Offer")
        _shot(page, "06-explore-offer.png")
        try:
            _tab(page, "Context")
            _shot(page, "06-explore-offer-context.png")
        except Exception as exc:
            print(f"skip offer Context: {exc}")
        try:
            _tab(page, "Chat")
            page.wait_for_timeout(2500)
            _shot(page, "07-explore-chat.png")
        except Exception as exc:
            print(f"skip Explore Chat: {exc}")

        _nav(page, "Stats")
        page.get_by_text("Five focused lenses", exact=False).first.wait_for(
            timeout=60_000
        )
        page.wait_for_timeout(4000)
        try:
            _tab(page, "Overview")
            page.wait_for_timeout(2000)
        except Exception as exc:
            print(f"skip Stats Overview tab: {exc}")
        _shot(page, "08-stats.png")
        try:
            page.get_by_role("tab", name="Market").click()
            page.wait_for_timeout(3000)
            _shot(page, "08b-stats-market.png")
        except Exception as exc:
            print(f"skip Stats Market: {exc}")
        try:
            page.get_by_role("tab", name="Livability").click()
            page.wait_for_timeout(4000)
            _shot(page, "08c-stats-livability.png")
        except Exception as exc:
            print(f"skip Stats Livability: {exc}")
        try:
            page.get_by_role("tab", name="Model").click()
            page.wait_for_timeout(8000)
            _shot(page, "08d-stats-model.png")
        except Exception as exc:
            print(f"skip Stats Model: {exc}")

        _nav(page, "Admin")
        page.wait_for_timeout(3000)
        _shot(page, "09-admin.png")
        try:
            page.get_by_text("Stats", exact=True).last.click()
            page.wait_for_timeout(2500)
            _shot(page, "10-admin-settings.png")
        except Exception as exc:
            print(f"skip Admin Stats section: {exc}")
            try:
                page.get_by_text("Offer AI", exact=True).click()
                page.wait_for_timeout(2500)
                _shot(page, "10-admin-settings.png")
            except Exception as exc2:
                print(f"skip Admin Offer AI: {exc2}")

        browser.close()
    print("done:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
