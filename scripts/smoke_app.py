"""Headless smoke test of the Streamlit app (no browser): runs the script through key interactions
and fails on any uncaught exception.    python scripts/smoke_app.py"""
from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def by_label(items, label):
    return next(x for x in items if x.label == label)


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
    by_label(at.toggle, "Extrapolate in k").set_value(True).run(); check(at, "extrapolation on")
    by_label(at.toggle, "Wireframe").set_value(True).run(); check(at, "wireframe on")
    by_label(at.radio, "Z axis").set_value("w").run(); check(at, "z = total variance")
    by_label(at.selectbox, "X axis").set_value("strike").run(); check(at, "x = strike")
    by_label(at.radio, "Method").set_value("svi").run(); check(at, "method = svi")
    by_label(at.radio, "Surface dynamics").set_value("compare").run(); check(at, "risk: compare dynamics")
    by_label(at.slider, "Time forward (calendar days)").set_value(70).run(); check(at, "risk: horizon beyond expiry")
    by_label(at.toggle, "3D P&L surface").set_value(True).run(); check(at, "risk: 3D P&L")
    by_label(at.button, "Reset camera").click().run(); check(at, "reset camera")
    at.sidebar.radio[0].set_value("synthetic · dirty").run(); check(at, "dirty fixture")
    at.sidebar.radio[0].set_value("synthetic · seed").run(); check(at, "generated synthetic seed 7")
    at.sidebar.radio[0].set_value("import csv").run(); check(at, "import mode without file (no demo substitution)")
    assert any("never substituted" in i.value for i in at.info), "import empty state missing"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
