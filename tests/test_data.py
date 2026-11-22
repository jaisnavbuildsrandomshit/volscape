from pathlib import Path

import pandas as pd

from volscape.data import TEMPLATE_CSV, ValidationSettings, load_and_validate
from volscape.synthetic import SyntheticSpec, generate_chain

ROOT = Path(__file__).resolve().parents[1]


def _reasons(res):
    return dict(zip(res.rejected.csv_line, res.rejected.reasons))


def test_clean_fixture_fully_accepted():
    b = (ROOT / "tests/fixtures/clean_chain.csv").read_bytes()
    res = load_and_validate(b, "clean", True)
    assert res.ok and len(res.rejected) == 0 and len(res.accepted) == res.n_input
    assert res.meta.underlying == "SYNTH-IDX" and res.meta.is_synthetic
    assert res.meta.valuation_time == pd.Timestamp("2026-06-15T20:00:00Z")


def test_bundled_fixture_matches_generator():
    # the committed clean fixture is exactly reproducible from the disclosed generator + seed
    disk = pd.read_csv(ROOT / "tests/fixtures/clean_chain.csv")
    gen = generate_chain(SyntheticSpec(seed=7, iv_noise_volpts=0.0))
    pd.testing.assert_frame_equal(disk, gen.reset_index(drop=True), check_dtype=False)


def test_dirty_fixture_row_level_reasons():
    b = (ROOT / "data/synthetic_dirty.csv").read_bytes()
    res = load_and_validate(b, "dirty", True)
    assert res.ok
    r = _reasons(res)
    expect = {5: "bid: missing", 10: "lacks timezone", 14: "strike must be > 0", 19: "ask < bid",
              23: "option_type", 28: "spot", 32: "expired", 37: "non-numeric", 42: "rate",
              46: "above model-free upper bound", 57: "American"}
    for line, frag in expect.items():
        assert line in r and frag in r[line], (line, r.get(line))
    # both copies of the duplicate are rejected (no authoritative choice is made)
    dup_lines = [ln for ln, why in r.items() if "duplicate contract" in why]
    assert len(dup_lines) == 2
    # wide spread is flagged, not rejected
    assert 52 not in r
    assert "WIDE_SPREAD" in res.accepted.loc[res.accepted.csv_line == 52, "flags"].iloc[0]
    assert len(res.accepted) + len(res.rejected) == res.n_input


def test_american_requires_explicit_convention():
    b = (ROOT / "data/synthetic_dirty.csv").read_bytes()
    res = load_and_validate(b, "dirty", True, ValidationSettings(allow_american_as_european=True))
    assert 57 not in _reasons(res)
    assert "AMERICAN_TREATED_AS_EUROPEAN" in res.accepted.loc[res.accepted.csv_line == 57, "flags"].iloc[0]
    assert any("American" in w for w in res.warnings)


def test_bad_uploads_fail_gracefully():
    for content in (b"\x00\xff\x10garbage", b"", b"a,b,c\n1,2,3\n", TEMPLATE_CSV.splitlines()[0].encode()):
        res = load_and_validate(content, "upload", False)
        assert not res.ok and res.fatal and res.accepted.empty


def test_template_round_trips_and_naive_timestamps_rejected():
    res = load_and_validate(TEMPLATE_CSV.encode(), "template", False)
    assert res.ok and len(res.accepted) == 2 and not res.meta.is_synthetic
    naive = TEMPLATE_CSV.replace("+00:00", "")
    res2 = load_and_validate(naive.encode(), "naive", False)
    assert not res2.ok and all("timezone" in x for x in res2.rejected.reasons)
