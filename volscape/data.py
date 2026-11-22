"""Option-chain CSV loading and validation.

Principle: never silently repair or invent observations. Every input row is either accepted
(possibly with warning flags) or rejected with explicit, row-level reasons. Fatal problems
(unreadable file, missing required columns, no consistent snapshot) produce a failed
ValidationResult instead of an exception, so the UI never crashes or silently swaps data.

Normalisations that ARE applied (documented, not repairs): option_type accepts C/P/call/put
(case-insensitive) and is stored as "C"/"P"; whitespace around fields is stripped;
timestamps are converted to UTC after parsing their explicit offset.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .conventions import SECONDS_PER_YEAR, MODEL_CONVENTION
from .pricing import price_bounds

REQUIRED_COLUMNS = ["valuation_time", "expiry", "strike", "option_type", "bid", "ask",
                    "spot", "rate", "dividend_yield"]
OPTIONAL_COLUMNS = ["underlying", "currency", "volume", "open_interest", "exercise_style"]
NUMERIC_COLUMNS = ["strike", "bid", "ask", "spot", "rate", "dividend_yield"]

TEMPLATE_CSV = (
    ",".join(REQUIRED_COLUMNS + ["underlying", "currency", "volume", "open_interest"]) + "\n"
    "2026-06-15T20:00:00+00:00,2026-07-17T20:00:00+00:00,100,C,2.41,2.47,100.0,0.04,0.015,"
    "EXAMPLE,USD,120,900\n"
    "2026-06-15T20:00:00+00:00,2026-07-17T20:00:00+00:00,100,P,2.10,2.16,100.0,0.04,0.015,"
    "EXAMPLE,USD,95,850\n"
)


@dataclass
class ValidationSettings:
    max_rel_spread: float = 0.60          # (ask-bid)/mid above this -> WIDE_SPREAD flag
    reject_wide_spreads: bool = False
    allow_american_as_european: bool = False
    bound_tol: float = 1e-8

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class SnapshotMeta:
    source_label: str
    is_synthetic: bool
    valuation_time: pd.Timestamp | None = None
    underlying: str = "UNSPECIFIED"
    currency: str = "UNSPECIFIED"
    spot: float | None = None
    input_sha256: str = ""
    model_convention: str = MODEL_CONVENTION
    exercise_note: str = "European exercise assumed (BSM)."

    def to_dict(self):
        d = dict(self.__dict__)
        d["valuation_time"] = None if self.valuation_time is None else self.valuation_time.isoformat()
        d["data_kind"] = "SYNTHETIC" if self.is_synthetic else "IMPORTED"
        return d


@dataclass
class ValidationResult:
    ok: bool
    meta: SnapshotMeta
    accepted: pd.DataFrame
    rejected: pd.DataFrame
    n_input: int
    fatal: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _parse_ts(s):
    """Parse one timestamp; returns (Timestamp UTC | None, error | None). Requires an offset."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return None, "missing"
    try:
        ts = pd.Timestamp(str(s).strip())
    except (ValueError, TypeError):
        return None, "unparseable timestamp"
    if ts is pd.NaT:
        return None, "unparseable timestamp"
    if ts.tzinfo is None:
        return None, "timestamp lacks timezone offset"
    return ts.tz_convert("UTC"), None


def _fail(meta, msg, n=0, raw=None):
    rej = pd.DataFrame(columns=["csv_line", "reasons"])
    return ValidationResult(False, meta, pd.DataFrame(), rej, n, fatal=[msg])


def load_and_validate(content: bytes, source_label: str, is_synthetic: bool,
                      settings: ValidationSettings | None = None) -> ValidationResult:
    settings = settings or ValidationSettings()
    meta = SnapshotMeta(source_label=source_label, is_synthetic=is_synthetic,
                        input_sha256=sha256_bytes(content))
    try:
        raw = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, skipinitialspace=True)
    except Exception as exc:  # noqa: BLE001 - any parse failure must be reported, not raised
        return _fail(meta, f"Could not read CSV: {type(exc).__name__}: {exc}")
    return validate_chain(raw, meta, settings)


def validate_chain(raw: pd.DataFrame, meta: SnapshotMeta, settings: ValidationSettings) -> ValidationResult:
    raw = raw.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    n = len(raw)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        return _fail(meta, f"Missing required column(s): {', '.join(missing_cols)}", n)
    if n == 0:
        return _fail(meta, "CSV has a header but no data rows", n)
    raw = raw.apply(lambda col: col.astype(str).str.strip())

    reasons: list[list[str]] = [[] for _ in range(n)]
    flags: list[list[str]] = [[] for _ in range(n)]
    df = pd.DataFrame(index=raw.index)
    df["csv_line"] = raw.index + 2  # header is line 1

    for col in ("valuation_time", "expiry"):
        vals = []
        for i, s in enumerate(raw[col]):
            ts, err = _parse_ts(s)
            if err:
                reasons[i].append(f"{col}: {err}")
            vals.append(ts)
        df[col] = pd.Series(vals, index=raw.index, dtype="object")

    for col in NUMERIC_COLUMNS:
        num = pd.to_numeric(raw[col].replace("", np.nan), errors="coerce")
        for i in np.where(~np.isfinite(num.to_numpy(float)))[0]:
            reasons[i].append(f"{col}: missing" if raw[col].iloc[i] in ("", "nan") else
                              f"{col}: non-numeric or nonfinite '{raw[col].iloc[i]}'")
        df[col] = num.astype(float)

    ot = raw["option_type"].str.upper().map({"C": "C", "CALL": "C", "P": "P", "PUT": "P"})
    for i in np.where(ot.isna())[0]:
        reasons[i].append(f"option_type: '{raw['option_type'].iloc[i]}' not one of C/P/call/put")
    df["option_type"] = ot

    for col in ("underlying", "currency"):
        df[col] = raw[col] if col in raw.columns else ""
    for col in ("volume", "open_interest"):
        df[col] = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else np.nan

    style = raw["exercise_style"].str.lower() if "exercise_style" in raw.columns else pd.Series("", index=raw.index)
    for i, s in enumerate(style):
        if s in ("", "european", "e"):
            continue
        if s in ("american", "a"):
            if settings.allow_american_as_european:
                flags[i].append("AMERICAN_TREATED_AS_EUROPEAN")
            else:
                reasons[i].append("exercise_style: American - European BSM would misprice early "
                                  "exercise; enable explicit European approximation to include")
        else:
            reasons[i].append(f"exercise_style: unknown '{s}'")

    # ---- per-row economic checks (only where fields parsed) ----
    v = df
    for i in range(n):
        r = v.iloc[i]
        if np.isfinite(r.spot) and r.spot <= 0:
            reasons[i].append("spot must be > 0")
        if np.isfinite(r.strike) and r.strike <= 0:
            reasons[i].append("strike must be > 0")
        if np.isfinite(r.bid) and r.bid < 0:
            reasons[i].append("bid must be >= 0")
        if np.isfinite(r.ask) and np.isfinite(r.bid) and r.ask < r.bid:
            reasons[i].append("ask < bid (crossed quote)")
        if np.isfinite(r.ask) and r.ask <= 0:
            reasons[i].append("ask must be > 0 (no offer)")
        if r.valuation_time is not None and r.expiry is not None and r.expiry <= r.valuation_time:
            reasons[i].append("expired: expiry <= valuation_time")

    # ---- snapshot consistency ----
    def modal(series):
        s = series.dropna()
        if s.empty:
            return None, 0.0
        vc = s.value_counts()
        return vc.index[0], vc.iloc[0] / len(series)

    vt_mode, vt_share = modal(df["valuation_time"].map(lambda t: None if t is None else t.isoformat()))
    spot_mode, spot_share = modal(df["spot"].where(np.isfinite(df["spot"])))
    if vt_mode is None or vt_share < 0.5:
        return _fail(meta, "No single valuation_time covers at least half the rows; a snapshot "
                           "must have one valuation time", n)
    if spot_mode is None or spot_share < 0.5:
        return _fail(meta, "No single spot value covers at least half the rows; a snapshot "
                           "must have one spot", n)
    vt_mode_ts = pd.Timestamp(vt_mode)
    for i in range(n):
        t = df["valuation_time"].iloc[i]
        if t is not None and t != vt_mode_ts:
            reasons[i].append(f"valuation_time {t.isoformat()} inconsistent with snapshot {vt_mode}")
        s = df["spot"].iloc[i]
        if np.isfinite(s) and s != spot_mode:
            reasons[i].append(f"spot {s:g} inconsistent with snapshot spot {spot_mode:g}")
    for col in ("underlying", "currency"):
        vals = df[col][df[col] != ""]
        if not vals.empty:
            m = vals.value_counts().index[0]
            for i in np.where((df[col] != "") & (df[col] != m))[0]:
                reasons[i].append(f"{col} '{df[col].iloc[i]}' inconsistent with snapshot '{m}'")
            setattr(meta, col, m)

    # rate / dividend must be single-valued within each expiry (term structure across expiries allowed)
    exp_key = df["expiry"].map(lambda t: None if t is None else t.isoformat())
    for col in ("rate", "dividend_yield"):
        tmp = pd.DataFrame({"e": exp_key, "v": df[col]}).dropna()
        for e, grp in tmp.groupby("e"):
            if grp["v"].nunique() > 1:
                m = grp["v"].value_counts().index[0]
                for i in grp.index[grp["v"] != m]:
                    reasons[i].append(f"{col} {df[col].iloc[i]:g} inconsistent with {m:g} for expiry {e}")

    # duplicates: same contract key -> reject every copy (cannot choose an authoritative quote)
    key = exp_key.astype(str) + "|" + df["strike"].astype(str) + "|" + df["option_type"].astype(str)
    dup = key.duplicated(keep=False) & exp_key.notna() & df["option_type"].notna() & np.isfinite(df["strike"])
    for i in np.where(dup)[0]:
        lines = df["csv_line"][key == key.iloc[i]].tolist()
        reasons[i].append(f"duplicate contract (expiry, strike, type) on lines {lines}")

    # ---- derived quantities and model bounds for rows still clean ----
    df["T"] = [((e - vt).total_seconds() / SECONDS_PER_YEAR) if (e is not None and vt is not None) else np.nan
               for e, vt in zip(df["expiry"], df["valuation_time"])]
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    clean = np.array([len(rs) == 0 for rs in reasons])
    if clean.any():
        c = df[clean]
        lo, hi = price_bounds(c.spot, c.strike, c["T"], c.rate, c.dividend_yield, c.option_type == "C")
        for j, i in enumerate(np.where(clean)[0]):
            r = df.iloc[i]
            if r.ask < lo[j] - settings.bound_tol:
                reasons[i].append(f"ask {r.ask:g} below model-free lower bound {lo[j]:.4f}")
            if r.bid > hi[j] + settings.bound_tol:
                reasons[i].append(f"bid {r.bid:g} above model-free upper bound {hi[j]:.4f}")
            if r.mid < lo[j] - settings.bound_tol <= r.ask:
                # not a tradable violation (ask still above the bound) but mid IV is undefined
                flags[i].append("MID_BELOW_BOUND")
            if r.bid == 0:
                flags[i].append("ZERO_BID")
            rel = (r.ask - r.bid) / r.mid if r.mid > 0 else np.inf
            if rel > settings.max_rel_spread:
                if settings.reject_wide_spreads:
                    reasons[i].append(f"relative spread {rel:.0%} exceeds {settings.max_rel_spread:.0%}")
                else:
                    flags[i].append("WIDE_SPREAD")

    df["reasons"] = ["; ".join(rs) for rs in reasons]
    df["flags"] = [";".join(fs) for fs in flags]
    rej_mask = df["reasons"] != ""
    rejected = raw[rej_mask].copy()
    rejected.insert(0, "csv_line", df.loc[rej_mask, "csv_line"])
    rejected["reasons"] = df.loc[rej_mask, "reasons"]
    rejected = rejected.reset_index(drop=True)

    acc = df[~rej_mask].copy()
    warnings = []
    if acc.empty:
        res = ValidationResult(False, meta, acc, rejected, n, fatal=["All rows were rejected"])
        return res
    acc["valuation_time"] = pd.to_datetime(acc["valuation_time"].map(lambda t: t.isoformat()), utc=True)
    acc["expiry"] = pd.to_datetime(acc["expiry"].map(lambda t: t.isoformat()), utc=True)
    acc["forward"] = acc.spot * np.exp((acc.rate - acc.dividend_yield) * acc["T"])
    acc["k"] = np.log(acc.strike / acc.forward)
    acc["rel_spread"] = (acc.ask - acc.bid) / acc.mid
    acc = acc.drop(columns=["reasons"]).sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)
    meta.valuation_time = vt_mode_ts
    meta.spot = float(spot_mode)
    if (acc["flags"].str.contains("AMERICAN")).any():
        meta.exercise_note = ("WARNING: some American-style contracts are priced with the European "
                              "BSM approximation (explicit user setting); early-exercise premium ignored.")
        warnings.append(meta.exercise_note)
    if meta.underlying == "UNSPECIFIED":
        warnings.append("No underlying column; underlying label is UNSPECIFIED.")
    if meta.currency == "UNSPECIFIED":
        warnings.append("No currency column; currency is UNSPECIFIED.")
    return ValidationResult(True, meta, acc, rejected, n, warnings=warnings)
