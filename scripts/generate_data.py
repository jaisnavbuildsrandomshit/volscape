"""Regenerate the bundled SYNTHETIC datasets deterministically.

    python scripts/generate_data.py

Writes:
  data/synthetic_demo.csv        seed 7, 0.15 vol-pt IV noise (demo dataset)
  data/synthetic_dirty.csv       clean subset + injected defects (validation demo)
  data/synthetic_dirty_NOTES.txt line-by-line description of the injected defects
  data/example_chain.csv         small 3-expiry example in the import schema
  tests/fixtures/clean_chain.csv seed 7, zero IV noise (numerical test fixture)
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from volscape.synthetic import SyntheticSpec, generate_chain, make_dirty_chain  # noqa: E402

DEMO_SPEC = SyntheticSpec(seed=7, iv_noise_volpts=0.15)
CLEAN_SPEC = SyntheticSpec(seed=7, iv_noise_volpts=0.0)


def main():
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    demo = generate_chain(DEMO_SPEC)
    demo.to_csv(ROOT / "data" / "synthetic_demo.csv", index=False)
    clean = generate_chain(CLEAN_SPEC)
    clean.to_csv(ROOT / "tests" / "fixtures" / "clean_chain.csv", index=False)
    dirty, notes = make_dirty_chain(CLEAN_SPEC)
    dirty.to_csv(ROOT / "data" / "synthetic_dirty.csv", index=False)
    (ROOT / "data" / "synthetic_dirty_NOTES.txt").write_text(
        "SYNTHETIC dirty fixture - deliberately injected defects (CSV line numbers, header = line 1)\n"
        + "\n".join(notes) + "\n", encoding="utf-8")
    exps = sorted(demo["expiry"].unique())
    ex = demo[demo["expiry"].isin([exps[2], exps[4], exps[8]])].copy()
    ex["underlying"] = "EXAMPLE"
    ex.to_csv(ROOT / "data" / "example_chain.csv", index=False)
    print(f"demo {len(demo)} rows, clean {len(clean)}, dirty {len(dirty)}, example {len(ex)}")


if __name__ == "__main__":
    main()
