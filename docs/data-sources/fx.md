# NBP FX mid-rates

| | |
|--|--|
| **Catalog key** | `fx` |
| **Module** | `src/happy_housing/ingest/fx.py` |
| **URL** | https://api.nbp.pl/ (table A) |
| **Artifact** | `data/processed/fx_nbp_cache.json` |

## Why we download it

Convert non-PLN Otodom asking prices to PLN before modeling and UI.

## What we extract

Per currency: mid rate, effective date → listing `price_original`,
`price_currency`, `price_pln`, `fx_rate_to_pln`, recomputed `price_per_m2`.

## How download / increment works

Lazy fetch when a non-PLN currency appears; daily JSON cache; PLN passthrough.

## Volume

Tiny (one rate per currency per day).

## Completeness

**Complete** for currencies observed in listings (not a full NBP archive).
