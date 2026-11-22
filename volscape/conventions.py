"""Units and day-count conventions used throughout VOLSCAPE.

* Time to expiry T is in years, ACT/365 (fixed): T = (expiry - valuation) seconds / (365 * 86400).
* Rates r and dividend yield q are continuously compounded, annualised decimals (0.04 = 4 %).
* Volatility sigma is an annualised decimal (0.20 = 20 vol points).
* Prices are per one unit of underlying, in the snapshot currency. Contract multipliers are
  applied only in the portfolio / scenario layer.
"""
from __future__ import annotations

import pandas as pd

DAYS_PER_YEAR = 365.0
SECONDS_PER_YEAR = DAYS_PER_YEAR * 86400.0

MODEL_CONVENTION = (
    "European Black-Scholes-Merton, continuous r and q, ACT/365 fixed, "
    "F = S·exp((r−q)T), k = ln(K/F), w = IV²·T"
)


def year_fraction(valuation: pd.Timestamp, expiry: pd.Timestamp) -> float:
    """ACT/365 fixed year fraction between two timezone-aware timestamps."""
    if valuation.tzinfo is None or expiry.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return (expiry - valuation).total_seconds() / SECONDS_PER_YEAR
