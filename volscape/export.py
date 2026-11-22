"""Reproducible exports: machine-readable experiment JSON and tabular CSVs."""
from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import __version__


def dependency_versions() -> dict:
    import numpy, scipy, pandas, plotly  # noqa: E401
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "volscape": __version__,
           "numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
           "plotly": plotly.__version__}
    try:
        import streamlit
        out["streamlit"] = streamlit.__version__
    except ImportError:
        pass
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not math.isfinite(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def frame_records(df: pd.DataFrame) -> list:
    return _clean(df.to_dict(orient="records")) if df is not None and len(df) else []


def surface_grid_frame(model, n_k=61, n_t=40) -> pd.DataFrame:
    """Long-format grid. 'extrapolated' marks k-extrapolated bands (only present when enabled)."""
    parts = []
    for name, (KK, TT, IV) in model.grid(n_k, n_t).items():
        parts.append(pd.DataFrame({"T_years": TT.ravel(), "days": TT.ravel() * 365, "k": KK.ravel(),
                                   "iv": IV.ravel(), "total_variance": (IV ** 2 * TT).ravel(),
                                   "extrapolated": name != "core", "region": name}))
    return pd.concat(parts, ignore_index=True).sort_values(["T_years", "k"]).reset_index(drop=True)


def experiment_record(res, method: str, seed=None, synthetic_spec=None) -> dict:
    diag = res.diagnostics(method)
    v = res.validation
    rec = {
        "schema": "volscape.experiment/1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {**v.meta.to_dict(), "n_input_rows": v.n_input, "n_accepted": len(v.accepted),
                 "n_rejected": len(v.rejected)},
        "seed": seed,
        "synthetic_generator": synthetic_spec,
        "settings": {**res.settings.to_dict(), "surface_method": method},
        "dependencies": dependency_versions(),
        "results": {
            "headline": diag["headline"],
            "aggregate_errors_fit_points": diag["aggregate"],
            "aggregate_errors_all_quotes": diag["aggregate_all"],
            "per_expiry_errors": frame_records(diag["per_expiry"]),
            "svi_slices": [f.to_row() for f in res.fits],
            "calendar_check": frame_records(diag["calendar"]),
            "butterfly_check": frame_records(diag["butterfly"]),
            "call_price_checks": frame_records(diag["call_checks"]),
            "iv_status_counts": {side: res.chain[f"iv_{side}_status"].value_counts().to_dict()
                                 for side in ("bid", "mid", "ask")},
            "rejections": frame_records(v.rejected[["csv_line", "reasons"]]) if len(v.rejected) else [],
            "messages": res.messages,
            "finite_grid_note": "Finite-grid checks detect sampled violations only; they do not prove "
                                "the global absence of arbitrage.",
        },
    }
    return _clean(rec)


def to_json(rec: dict) -> str:
    return json.dumps(rec, indent=2, sort_keys=False, allow_nan=False)
