"""Home About tab copy — business purpose in plain language."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AboutSection:
    """One titled block on the Home About tab."""

    title: str
    """Short section heading."""
    body: str
    """Markdown body (short paragraphs, no jargon stack)."""


ABOUT_SECTIONS: tuple[AboutSection, ...] = (
    AboutSection(
        title="What this is",
        body=(
            "happy-housing helps you read Warsaw flat prices with more context "
            "than a listing site alone. For each offer it estimates a fair "
            "**asking price per m²**, compares that to the listed price, and "
            "shows nearby factors that often move value — transit, noise, air, "
            "commute, parks, and the like.\n\n"
            "It runs on your machine from public data. No paid valuation API."
        ),
    ),
    AboutSection(
        title="Who it is for",
        body=(
            "Buyers who want a second opinion before calling an agent. "
            "Investors scanning districts for asks that look soft or stiff "
            "versus recent sales. Anyone curious how location shows up in "
            "price — you do not need a data background to use the screens."
        ),
    ),
    AboutSection(
        title="What we model",
        body=(
            "The core number is **model zł/m²**: given this flat and this pin "
            "on the map, what asking level looks typical in our training data. "
            "**Delta** is listing zł/m² minus that model value. "
            "Negative delta → ask sits below the model; positive → above.\n\n"
            "We train on property basics (size, rooms, year, amenities when "
            "known) plus location signals: distances to transit and amenities, "
            "aircraft noise bands around Chopin, nearby air quality, "
            "construction permits, and commute time to a workplace you can "
            "set. Sale prices from Warsaw's Rejestr Cen (RCN) sit beside "
            "listing asks so you can see ask versus what sold nearby.\n\n"
            "A separate preference score can rank offers against your starred "
            "likes — that is not the price model; it is a fit helper."
        ),
    ),
    AboutSection(
        title="Explore Assistant",
        body=(
            "On **Explore**, open the **Chat** tab or the **Assistant** "
            "popover. It is a local-model chat (LM Studio) that calls tools "
            "over your scored listings — it does not invent ids or prices.\n\n"
            "Ask it to search and recommend flats, update Explore preferences "
            "in plain language, read starred watches and their news, pull "
            "cached offer AI reviews, look up public web opinions, or open "
            "an offer in the UI. Conversations persist so you can continue "
            "later.\n\n"
            "Admin → Assistant turns the chat on/off and tunes the prompt. "
            "Offer-page **AI review** is a separate one-shot write-up, not "
            "the multi-turn Assistant."
        ),
    ),
    AboutSection(
        title="What it is not",
        body=(
            "Not a bank appraisal, not a notary opinion, and not investment "
            "advice. Listings go stale; public layers have gaps; the model "
            "can be wrong on odd flats. Treat it as a research lens — then "
            "verify on the street and with people you trust."
        ),
    ),
)
"""Ordered About sections shown as the default Home tab."""


def about_section_titles() -> tuple[str, ...]:
    """Stable titles for tests and docs."""
    return tuple(section.title for section in ABOUT_SECTIONS)
