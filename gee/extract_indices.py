"""
Extract monthly large-scale ocean climate indices from Google Earth Engine.

WHY
---
Notebook 06 showed that local, lagged reanalysis variables carry little
information beyond persistence at a one-month lead. Pakistan's temperature
and monsoon are known to respond to remote sea-surface temperature (SST)
patterns, so we add three standard SST indices as candidate predictors:

  nino34    El Nino-Southern Oscillation (ENSO): mean SST anomaly in the
            Nino-3.4 box (5S-5N, 170W-120W). Positive = El Nino.
  dmi       Indian Ocean Dipole Mode Index: SST anomaly of the western box
            (10S-10N, 50E-70E) minus the south-eastern box (10S-0, 90E-110E).
            Positive = positive IOD (warm west Indian Ocean).
  trop_sst  mean SST anomaly over the global tropical ocean (20S-20N).
            Tracks the background warming of the oceans.

DATA
----
NOAA OISST v2.1 daily (`NOAA/CDR/OISST/V2_1`, 0.25 deg, Sep 1981 - present),
averaged to calendar months.

ANOMALY BASE PERIOD: 1982-2010
------------------------------
The anomalies use a climatology from 1982-2010, which ends BEFORE every
validation and test year (the earliest CV validation block starts in 2011).
So the index anomalies contain no information from any evaluation period.
This is also how operational centres compute ENSO indices (a fixed historical
base period).

USAGE (from the project folder)
    python gee/extract_indices.py --check
    python gee/extract_indices.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from gee.extract_data import call_with_retry, init_ee  # noqa: E402

OISST = "NOAA/CDR/OISST/V2_1"
FIRST_YEAR = 1982
BASE_YEARS = (1982, 2010)
SCALE_M = 27830          # native 0.25 degree grid

# Longitudes east of 180 are written as negative degrees (170W = -170).
BOXES = {                       # [west, south, east, north]
    "nino34": [-170, -5, -120, 5],
    "iod_west": [50, -10, 70, 10],
    "iod_east": [90, -10, 110, 0],
    "tropics": [-180, -20, 180, 20],
}


def box(name: str) -> ee.Geometry:
    # geodesic=False: a plain lat/lon rectangle, as used in the index definitions
    return ee.Geometry.Rectangle(BOXES[name], None, False)


def sst_band() -> tuple[str, float]:
    """Return the SST band name and its scale factor (OISST stores 0.01 deg C integers)."""
    img = ee.ImageCollection(OISST).first()
    bands = img.bandNames().getInfo()
    band = "sst" if "sst" in bands else bands[0]
    return band, 0.01


def run_check() -> None:
    coll = ee.ImageCollection(OISST)
    bands = coll.first().bandNames().getInfo()
    r = coll.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
    print("Bands:", bands)
    print("Date range:", ee.Date(r.get("min")).format("YYYY-MM-dd").getInfo(), "->",
          ee.Date(r.get("max")).format("YYYY-MM-dd").getInfo())
    band, scale = sst_band()
    v = coll.filterDate("2015-12-01", "2016-01-01").select(band).mean().multiply(scale) \
            .reduceRegion(ee.Reducer.mean(), box("nino34"), SCALE_M).getInfo()
    print(f"Mean Nino-3.4 SST in Dec 2015 (strong El Nino; expect about 29 C): {v}")


def extract(band: str, scale: float) -> pd.DataFrame:
    def one_month(ym):
        ym = ee.List(ym)
        start = ee.Date.fromYMD(ym.get(0), ym.get(1), 1)
        img = ee.ImageCollection(OISST).filterDate(start, start.advance(1, "month")).select(band).mean().multiply(scale)
        # The global tropics box is huge; a ~1 degree grid is plenty for a basin-wide mean.
        vals = {n: img.reduceRegion(ee.Reducer.mean(), box(n), 111000 if n == "tropics" else SCALE_M,
                                    maxPixels=1e9, bestEffort=True).get(band)
                for n in BOXES}
        return ee.Feature(None, {"year": ym.get(0), "month": ym.get(1), **vals})

    rows = []
    # Small requests (6 months x 4 boxes): larger ones hit "Too many concurrent aggregations"
    all_months = [[y, m] for y in range(FIRST_YEAR, C.END_YEAR + 1) for m in range(1, 13)]
    CH = 6
    for i in range(0, len(all_months), CH):
        chunk = ee.List(all_months[i:i + CH])
        fc = ee.FeatureCollection(chunk.map(one_month))
        data = call_with_retry(lambda: fc.getInfo(), f"months {all_months[i]}..")
        rows += [f["properties"] for f in data["features"]]
        if all_months[i][1] == 1:
            print(f"  {all_months[i][0]} done", flush=True)
    return pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)


def to_indices(raw: pd.DataFrame) -> pd.DataFrame:
    """Monthly anomalies vs the 1982-2010 base period, and the three indices."""
    df = raw.copy()
    base = df[df["year"].between(*BASE_YEARS)]
    for k in BOXES:
        clim = base.groupby("month")[k].mean()
        df[f"{k}_anom"] = df[k] - df["month"].map(clim)
    out = df[["year", "month"]].copy()
    out["nino34"] = df["nino34_anom"]
    out["dmi"] = df["iod_west_anom"] - df["iod_east_anom"]
    out["trop_sst"] = df["tropics_anom"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    init_ee(args.project)
    if args.check:
        run_check()
        return
    band, scale = sst_band()
    raw = extract(band, scale)
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    raw.to_csv(C.DATA_RAW / "sst_boxes_raw.csv", index=False)
    idx = to_indices(raw)
    idx.to_csv(C.DATA_RAW / "climate_indices.csv", index=False)
    print(f"Saved {len(idx)} months -> climate_indices.csv")


if __name__ == "__main__":
    main()
