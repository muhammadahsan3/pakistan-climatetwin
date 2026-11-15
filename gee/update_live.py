"""
Live update of the Pakistan ClimateTwin (the "real-time" part of the digital twin).

What it does, every time it runs
--------------------------------
1. Asks Earth Engine how recent each dataset is and finds the latest COMPLETE month
   available for both ERA5-Land and NOAA OISST (the forecast inputs).
2. Downloads every month after the 2000-2025 study period up to that month
   (district means, same method as gee/extract_data.py) and any missing ocean months.
3. Rainfall: CHIRPS is published about six weeks late. For a month CHIRPS does not have
   yet, rainfall is estimated from ERA5-Land precipitation, scaled per district and calendar
   month by the CHIRPS/ERA5 ratio of the 2000-2019 averages (bias correction). Each month is
   flagged with its rainfall source.
4. Rebuilds anomalies (against the fixed 2000-2019 climatology) and features exactly as in
   training, then:
   * FORECAST: predicts the month after the latest data month for every district;
   * VERIFICATION: re-creates the forecast that would have been issued for every month
     since January 2026 and compares it with what actually happened. These months were never
     used for training or tuning, so this is a genuine out-of-sample test that grows each month.
5. Writes the results to dashboard/data/ for the Streamlit app.

The model is the best configuration from notebook 07: Ridge regression with ocean drivers,
hyperparameters from best_params_ocean.json, trained on target months 2000-2025 only.
The training set is fixed on purpose, so the verification months stay unseen.

USAGE (from the project folder)
    python gee/update_live.py            # local: uses your Earth Engine login
In GitHub Actions the EE_SERVICE_ACCOUNT_KEY and EE_PROJECT environment variables are used
(see .github/workflows/update-live.yml and README section 7).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C, features as F, preprocessing as P  # noqa: E402
from gee.extract_data import KELVIN, call_with_retry, init_ee, load_districts  # noqa: E402
from gee import extract_indices as XI  # noqa: E402

DASH = ROOT / "dashboard" / "data"
LIVE_START = pd.Period(f"{C.END_YEAR + 1}-01", "M")
OCEAN_SET = F.FEATURES + F.OCEAN_FEATURES


# ---------------------------------------------------------------------------
# Data availability
# ---------------------------------------------------------------------------
def last_complete_month(collection: str) -> pd.Period:
    """Latest calendar month for which the collection has its final day."""
    coll = ee.ImageCollection(collection).filterDate(str((LIVE_START - 3).start_time.date()), "2100-01-01")
    last = pd.Timestamp(ee.Date(coll.aggregate_max("system:time_start")).format("YYYY-MM-dd").getInfo())
    p = last.to_period("M")
    return p if (last + pd.Timedelta(days=1)).day == 1 else p - 1


# ---------------------------------------------------------------------------
# Climate months after the study period
# ---------------------------------------------------------------------------
def month_image(p: pd.Period, with_chirps: bool, rain_band: str) -> ee.Image:
    start = ee.Date.fromYMD(p.year, p.month, 1)
    end = start.advance(1, "month")
    era = ee.ImageCollection(C.ERA5_DAILY).filterDate(start, end)
    bands = [
        era.select("temperature_2m_max").mean().subtract(KELVIN).rename("tmax_c"),
        era.select("temperature_2m_min").mean().subtract(KELVIN).rename("tmin_c"),
        era.select("temperature_2m").mean().subtract(KELVIN).rename("tmean_c"),
        era.select("volumetric_soil_water_layer_1").mean().rename("soil_moisture"),
        era.select("surface_net_solar_radiation_sum").mean().divide(1e6).rename("solar_mj"),
        era.select("total_precipitation_sum").sum().multiply(1000).rename("era5_rain_mm"),
    ]
    if with_chirps:
        bands.append(ee.ImageCollection(C.CHIRPS_DAILY).filterDate(start, end).select(rain_band).sum().rename("chirps_mm"))
    return ee.Image.cat(bands)


def extract_live_months(months: list[pd.Period], chirps_last: pd.Period) -> pd.DataFrame:
    districts = load_districts()
    slim = districts.map(lambda f: ee.Feature(f.geometry(), {"district_id": f.get(C.BOUNDARY_ID_FIELD)}))
    rows = []
    for p in months:
        img = month_image(p, p <= chirps_last, "precipitation")
        fc = img.reduceRegions(slim, ee.Reducer.mean(), scale=C.REDUCE_SCALE_M, tileScale=4) \
                .map(lambda f: ee.Feature(None, ee.Feature(f).toDictionary()))
        data = call_with_retry(lambda: fc.getInfo(), str(p))
        df = pd.DataFrame([f["properties"] for f in data["features"]])
        df["year"], df["month"] = p.year, p.month
        rows.append(df)
        print(f"  {p}: {len(df)} districts" + ("" if p <= chirps_last else "  (rainfall from bias-corrected ERA5-Land)"), flush=True)
    out = pd.concat(rows, ignore_index=True)
    if "chirps_mm" not in out:
        out["chirps_mm"] = np.nan
    return out


def fill_rainfall(live: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """CHIRPS where available; otherwise ERA5-Land rain x (CHIRPS / ERA5) 2000-2019 ratio."""
    base = hist[hist["year"].between(*C.CLIMATOLOGY_YEARS)]
    clim = base.groupby(["district_id", "month"])[["rain_mm", "era5_rain_mm"]].mean()
    ratio = (clim["rain_mm"] / clim["era5_rain_mm"].where(clim["era5_rain_mm"] > 1)).clip(0.2, 5).fillna(1.0)
    r = ratio.reindex(pd.MultiIndex.from_frame(live[["district_id", "month"]])).to_numpy()
    out = live.copy()
    has = out["chirps_mm"].notna()
    out["rain_mm"] = np.where(has, out["chirps_mm"], out["era5_rain_mm"] * r)
    out["rain_source"] = np.where(has, "CHIRPS v3", "ERA5-Land, bias-corrected")
    return out.drop(columns="chirps_mm")


# ---------------------------------------------------------------------------
# Ocean indices
# ---------------------------------------------------------------------------
def update_ocean(raw: pd.DataFrame, upto: pd.Period) -> pd.DataFrame:
    have = {pd.Period(f"{y}-{m:02d}", "M") for y, m in zip(raw["year"], raw["month"])}
    first = pd.Period(f"{XI.FIRST_YEAR}-01", "M")
    missing = [p for p in pd.period_range(first, upto, freq="M") if p not in have]
    if missing:
        print(f"Ocean indices: fetching {len(missing)} new month(s)")
        band, scale = XI.sst_band()
        new = XI.extract(band, scale, [[p.year, p.month] for p in missing])
        raw = pd.concat([raw, new], ignore_index=True).sort_values(["year", "month"]).reset_index(drop=True)
    return raw


# ---------------------------------------------------------------------------
# Forecast + verification
# ---------------------------------------------------------------------------
def forecast_and_verify(hist: pd.DataFrame, live: pd.DataFrame, idx_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    statics = hist.drop_duplicates("district_id")[["district_id", "district", "province", "elevation_m", "lat", "lon"]]
    live = live.drop(columns=[c for c in ["district", "province", "elevation_m", "lat", "lon"] if c in live]).merge(statics, on="district_id")
    cols = [c for c in hist.columns if c in live.columns]
    monthly = pd.concat([hist[cols], live[cols]], ignore_index=True)
    monthly = P.add_anomalies(P.add_time_columns(monthly))         # fixed 2000-2019 climatology

    feat = F.add_ocean_features(F.build_features(monthly, keep_unlabelled=True), F.load_indices(XI.to_indices(idx_raw)))
    feat = feat.dropna(subset=OCEAN_SET)
    train = feat[(feat["split"] != "forecast") & (feat["target_year"] <= C.END_YEAR)]
    new = feat[feat["target_year"] > C.END_YEAR]

    alpha = json.load(open(DASH / "best_params_ocean.json"))["Ridge"]["model__alpha"]
    model = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]).fit(train[OCEAN_SET], train[F.TARGET])
    phi = float((train["tmax_anom_l0"] * train[F.TARGET]).sum() / (train["tmax_anom_l0"] ** 2).sum())

    new = new.assign(pred_anom=model.predict(new[OCEAN_SET]), damped_anom=phi * new["tmax_anom_l0"])
    meta = json.load(open(DASH / "meta.json"))
    lo, hi = meta["interval_10_90"]

    fc = new[new["split"] == "forecast"].copy()
    fc["pred_tmax"] = fc["target_tmax_clim"] + fc["pred_anom"]
    fc["pred_tmax_lo"], fc["pred_tmax_hi"] = fc["pred_tmax"] + lo, fc["pred_tmax"] + hi
    fc = fc[["district_id", "district", "province", "input_date", "target_date", "tmax_anom_l0", "pred_anom",
             "damped_anom", "target_tmax_clim", "pred_tmax", "pred_tmax_lo", "pred_tmax_hi"]]

    ver = new[new["split"] != "forecast"][["district_id", "district", "province", "target_date", F.TARGET,
                                          "target_tmax", "target_tmax_clim", "pred_anom", "damped_anom"]].copy()
    summary = {}
    if len(ver):
        y = ver[F.TARGET]
        mse = lambda p: float(np.mean((p - y) ** 2))
        summary = {"months": int(ver["target_date"].nunique()),
                   "rmse_model": mse(ver["pred_anom"]) ** 0.5, "rmse_damped": mse(ver["damped_anom"]) ** 0.5,
                   "rmse_clim": mse(0 * y) ** 0.5,
                   "skill_vs_clim": 1 - mse(ver["pred_anom"]) / mse(0 * y),
                   "skill_vs_damped": 1 - mse(ver["pred_anom"]) / mse(ver["damped_anom"])}
    return fc, ver, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project")
    args = ap.parse_args()
    init_ee(args.project)

    era_last, chirps_last = last_complete_month(C.ERA5_DAILY), last_complete_month(C.CHIRPS_DAILY)
    sst_last = last_complete_month(XI.OISST)
    data_end = min(era_last, sst_last)
    print(f"Latest complete month - ERA5-Land: {era_last}, CHIRPS: {chirps_last}, OISST: {sst_last} -> using data up to {data_end}")
    if data_end < LIVE_START:
        print("No months after the study period yet; nothing to do.")
        return

    hist = pd.read_parquet(DASH / "history.parquet")
    months = list(pd.period_range(LIVE_START, data_end, freq="M"))
    print(f"Downloading {len(months)} month(s) of district climate data ...")
    live = fill_rainfall(extract_live_months(months, chirps_last), hist)

    idx_raw = update_ocean(pd.read_csv(DASH / "sst_boxes_raw.csv"), data_end)
    idx_raw.to_csv(DASH / "sst_boxes_raw.csv", index=False)

    fc, ver, summary = forecast_and_verify(hist, live, idx_raw)

    live_out = live.copy()
    live_out["date"] = pd.to_datetime(dict(year=live_out["year"], month=live_out["month"], day=1))
    live_out.to_parquet(DASH / "live_monthly.parquet", index=False)
    fc.to_csv(DASH / "forecast.csv", index=False)
    ver.to_parquet(DASH / "verification.parquet", index=False)

    meta = json.load(open(DASH / "meta.json"))
    meta.update({
        "live_data_end": str(data_end.start_time.date()),
        "chirps_data_end": str(chirps_last.start_time.date()),
        "forecast_month": str(fc["target_date"].iloc[0].date()),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "live_verification": summary,
    })
    json.dump(meta, open(DASH / "meta.json", "w"), indent=2)
    print(f"Forecast for {meta['forecast_month'][:7]}: national mean anomaly {fc['pred_anom'].mean():+.2f} °C")
    if summary:
        print(f"Live verification over {summary['months']} month(s): RMSE {summary['rmse_model']:.2f} °C "
              f"(climatology {summary['rmse_clim']:.2f}, damped persistence {summary['rmse_damped']:.2f}); "
              f"skill vs climatology {summary['skill_vs_clim']:+.1%}")


if __name__ == "__main__":
    main()
