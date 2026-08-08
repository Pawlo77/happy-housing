"""Home hero markup helpers (pure; Streamlit renders the HTML)."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True, slots=True)
class HeroStats:
    """Formatted strings for the Home brand strip."""

    offers_label: str
    r2_label: str
    refreshed_label: str


def format_hero_stats(
    refreshed: str | None,
    n_offers: int,
    r2: float | None,
) -> HeroStats:
    """Format headline metrics for the Home hero."""
    refreshed_label = "—"
    if refreshed:
        refreshed_label = refreshed[:19].replace("T", " ")
    r2_label = f"{r2:.3f}" if r2 is not None else "—"
    return HeroStats(
        offers_label=f"{n_offers:,}",
        r2_label=r2_label,
        refreshed_label=refreshed_label,
    )


def hero_html(stats: HeroStats) -> str:
    """Compact brand strip HTML — no card chrome, no badge pills."""
    offers = escape(stats.offers_label)
    r2 = escape(stats.r2_label)
    refreshed = escape(stats.refreshed_label)
    return f"""
<div class="hh-hero">
  <div class="hh-hero__brand">
    <p class="hh-hero__name">happy-housing</p>
    <p class="hh-hero__tag">Warsaw fair-price intelligence from open data</p>
  </div>
  <dl class="hh-hero__stats">
    <div>
      <dt>Scored offers</dt>
      <dd>{offers}</dd>
    </div>
    <div>
      <dt>Hold-out R²</dt>
      <dd>{r2}</dd>
    </div>
    <div>
      <dt>Last refresh (UTC)</dt>
      <dd class="hh-hero__muted">{refreshed}</dd>
    </div>
  </dl>
</div>
<style>
  .hh-hero {{
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.25rem 2rem;
    margin: 0 0 1.25rem 0;
    padding: 0.15rem 0 1.1rem 0;
    border-bottom: 1px solid rgba(24, 32, 29, 0.12);
  }}
  .hh-hero__name {{
    margin: 0;
    font-family: "Outfit", "IBM Plex Sans", sans-serif;
    font-size: clamp(1.85rem, 2.6vw, 2.35rem);
    font-weight: 650;
    letter-spacing: -0.035em;
    line-height: 1.05;
    color: #12181f;
  }}
  .hh-hero__tag {{
    margin: 0.35rem 0 0 0;
    max-width: 28rem;
    font-size: 0.98rem;
    line-height: 1.35;
    color: #5a6670;
  }}
  .hh-hero__stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 1.1rem 1.75rem;
    margin: 0;
  }}
  .hh-hero__stats div {{
    margin: 0;
  }}
  .hh-hero__stats dt {{
    margin: 0;
    font-size: 0.72rem;
    font-weight: 550;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #7a8792;
  }}
  .hh-hero__stats dd {{
    margin: 0.15rem 0 0 0;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 1.35rem;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
    color: #12181f;
  }}
  .hh-hero__muted {{
    font-size: 0.92rem !important;
    font-weight: 450 !important;
    color: #5a6670 !important;
  }}
</style>
"""
