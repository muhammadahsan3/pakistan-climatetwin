"""
Trend analysis for warming-hotspot identification.

KEY CONCEPTS
------------
Mann-Kendall (MK) test
    A NON-PARAMETRIC test for a monotonic trend. It compares every pair of years
    and counts how often the later value is higher. It needs no normality
    assumption and is robust to outliers, which is why it is the standard
    test in climate trend studies.
    H0: no monotonic trend.  A small p-value -> evidence of a trend.

Sen's slope
    The MEDIAN of the slopes between all pairs of years. It estimates how fast
    the variable changes (here reported in deg C per decade). Unlike an
    ordinary least-squares slope, one extreme year cannot drag it.

Multiple-testing correction (Benjamini-Hochberg FDR)
    We run one test per district (126 tests). At p < 0.05 we would expect about
    6 "significant" districts by pure chance even if nothing were changing.
    The Benjamini-Hochberg procedure converts p-values to q-values that control
    the False Discovery Rate: among districts called significant at q < 0.05,
    at most ~5% are expected to be false alarms. A district is called a
    warming hotspot only if q < 0.05 AND Sen's slope > 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy.stats import false_discovery_control

ALPHA = 0.05


def mk_sen(values: np.ndarray) -> dict:
    """Mann-Kendall test + Sen's slope for one series of annual values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    r = mk.original_test(v)
    return {"n_years": len(v), "tau": r.Tau, "p_value": r.p,
            "sen_slope_per_decade": r.slope * 10, "sen_intercept": r.intercept}


def district_trends(annual: pd.DataFrame, up: str = "Significant warming",
                    down: str = "Significant cooling") -> pd.DataFrame:
    """
    annual: district x year table (rows = district_id, columns = years).
    Returns one row per district with MK/Sen results, FDR q-values and a label.
    For non-temperature variables pass e.g. up="Significant increase", down="Significant decrease".
    """
    res = pd.DataFrame([mk_sen(row.values) | {"district_id": did} for did, row in annual.iterrows()])
    res["q_value"] = false_discovery_control(res["p_value"].values, method="bh")
    res["trend_class"] = np.select(
        [(res["q_value"] < ALPHA) & (res["sen_slope_per_decade"] > 0),
         (res["q_value"] < ALPHA) & (res["sen_slope_per_decade"] < 0),
         (res["p_value"] < ALPHA)],
        [up, down, "p<0.05 but not after FDR"],
        default="No significant trend")
    return res.set_index("district_id")


def national_trend(annual: pd.DataFrame) -> dict:
    """MK/Sen on the area-unweighted national mean series."""
    return mk_sen(annual.mean(axis=0).values)
