"""Home hero metric formatting (pure; Streamlit widgets render the UI)."""

from dataclasses import dataclass


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
