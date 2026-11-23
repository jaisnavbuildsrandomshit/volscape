"""3D figures must encode real data faithfully (no smoothing, no filled gaps)."""
import numpy as np
import pandas as pd

from volscape import charts as ch


def test_book_landscape_bins_quantities_and_keeps_gaps_empty():
    mid = 100.0
    hist = [
        (1000, mid, [(99.995, 2.0)], [(100.005, 3.0), (100.15, 1.0)]),   # -0.5 bp, +0.5 bp, +15 bp
        # second 1001 missing (e.g. reconnect) -> must stay empty
        (1002, mid, [(99.995, 4.0)], [(100.005, 1.0)]),
    ]
    fig = ch.book_landscape(hist, bins_bps=1.0, half_width_bps=20.0)
    s = fig.data[0]
    x, y, z = np.asarray(s.x), np.asarray(s.y), np.asarray(s.z, dtype=float)
    assert list(y) == [-2, -1, 0]
    assert np.isnan(z[1]).all()                          # missing second not interpolated
    jb, ja, j15 = (np.argmin(np.abs(x - v)) for v in (-0.5, 0.5, 15.5))
    assert z[0, jb] == 2.0 and z[0, ja] == 3.0 and z[0, j15] == 1.0
    assert z[2, jb] == 4.0 and z[2, ja] == 1.0
    assert np.nansum(z[0]) == 6.0                        # all displayed quantity accounted for


def test_density_waterfall_uses_log_moneyness_density_and_flags_negatives():
    exp = pd.Timestamp("2027-01-01", tz="UTC")
    d = pd.DataFrame({"strike": [90.0, 100.0, 110.0], "k": np.log([0.9, 1.0, 1.1]), "density": [0.01, -0.002, 0.01]})
    chain = pd.DataFrame({"expiry": [exp], "T": [0.5]})
    fig = ch.density_waterfall({exp: d}, chain)
    curtain = fig.data[0]
    assert np.allclose(np.asarray(curtain.z)[1], d.density * d.strike)   # q(k) = q(K)·K, no clipping
    assert any(getattr(t, "name", "") == "negative density" for t in fig.data)
