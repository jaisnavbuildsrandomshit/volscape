"""Headless smoke test of the Streamlit app (no browser, no network): runs the script through key
interactions and fails on any uncaught exception.    python scripts/smoke_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def by_label(items, label):
    return next(x for x in items if x.label == label)


def by_key(items, key):
    return next(x for x in items if x.key == key)


def check(at, step):
    exc = [e.value for e in at.exception]
    print(f"{step:<48} exceptions={len(exc)} errors={len(at.error)} warnings={len(at.warning)}")
    if exc:
        print("  ", exc[0])
        sys.exit(1)


def main():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    check(at, "initial load (bundled synthetic demo)")
    by_label(at.radio, "Method").set_value("baseline").run(); check(at, "method = baseline")
    by_key(at.button_group, "layers").set_value(["points", "slices", "contours", "wireframe", "extrapolate k"]).run()
    check(at, "layers: wireframe + extrapolation on")
    by_label(at.radio, "Z axis").set_value("w").run(); check(at, "z = total variance")
    by_label(at.selectbox, "X axis").set_value("strike").run(); check(at, "x = strike")
    by_label(at.radio, "Method").set_value("svi").run(); check(at, "method = svi")
    by_key(at.button_group, "layers").set_value([]).run(); check(at, "layers: all off")
    by_label(at.radio, "Surface dynamics").set_value("compare").run(); check(at, "risk: compare dynamics")
    by_label(at.slider, "Time forward (calendar days)").set_value(70).run(); check(at, "risk: horizon beyond expiry")
    by_label(at.toggle, "3D P&L surface").set_value(True).run(); check(at, "risk: 3D P&L")
    by_label(at.button, "Reset camera").click().run(); check(at, "reset camera")
    by_key(at.selectbox, "dataset").set_value("synthetic · dirty").run(); check(at, "dirty fixture")
    by_key(at.selectbox, "dataset").set_value("synthetic · seed").run(); check(at, "generated synthetic seed 7")
    import os
    for v in ("UPSTOX_ACCESS_TOKEN", "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
        os.environ.pop(v, None)                     # no credentials -> setup guidance, no network call
    os.environ["VOLSCAPE_NO_SECRETS_FILE"] = "1"   # never use a real local key file in the smoke run
    by_key(at.selectbox, "dataset").set_value("live · NSE (Upstox)").run(); check(at, "NSE without credentials (guidance)")
    assert any("UPSTOX_ACCESS_TOKEN" in i.value for i in at.info), "Upstox setup guidance missing"
    by_key(at.selectbox, "dataset").set_value("live · US (Alpaca)").run(); check(at, "US without credentials (guidance)")
    assert any("ALPACA_API_KEY_ID" in i.value for i in at.info), "Alpaca setup guidance missing"
    by_key(at.selectbox, "dataset").set_value("import csv").run(); check(at, "import mode without file (no demo substitution)")
    assert any("never substituted" in i.value for i in at.info), "import empty state missing"
    by_key(at.radio, "ws").set_value("LIVE").run(); check(at, "LIVE workspace (not connected, no network)")
    assert any("Not connected" in m.value for m in at.markdown), "LIVE must not auto-connect"
    by_key(at.radio, "ws").set_value("SURFACE").run(); check(at, "back to SURFACE")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
