"""
Feature engineering for next-month Tmax-anomaly prediction.

THE PREDICTION SETUP
--------------------
We stand at the END of month t (all of month t's data is known) and predict
month t+1:

    target  = Tmax anomaly of month t+1  (for district d)
    inputs  = information available at the end of month t

Every feature is computed from months <= t. The only exception is information
that is known in advance for ANY future month: which calendar month t+1 is,
the district's fixed attributes (elevation, location) and its climatology
(a long-term average computed from 2000-2019, available before 2020).

The train/validation/test split is done on the YEAR OF THE TARGET month.
So the first test row predicts Jan 2023 using Dec 2022 inputs. That is
legitimate: in reality Dec 2022 would be known when forecasting Jan 2023.

FEATURE GROUPS
--------------
1. Temperature memory : Tmax/Tmin anomalies at t, t-1, t-2 and their 3-month mean
2. Moisture memory    : rainfall anomaly at t, 3- and 6-month mean rainfall anomaly,
                        soil-moisture anomaly at t and t-1 (wet soil cools the next
                        month through evaporation, EDA finding F5/F7)
3. Radiation          : solar radiation anomaly at t
4. Large-scale context: national mean Tmax anomaly at t (heat waves and cool spells
                        affect many districts at once)
4b. Seasonal persistence: anomaly x sin/cos(target month) interactions
5. Seasonality        : target month as sin/cos (December and January are neighbours),
                        target-month Tmax climatology
6. Geography          : elevation, latitude, longitude

Deliberately NOT used: the calendar year. A year feature would let models
learn the trend in the training years, but trees cannot extrapolate it to
2023-2025, and it would make the linear model's behaviour depend on
extrapolation. Warming enters through the recent anomalies instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src import preprocessing as P

TARGET = "target_anom"

FEATURE_GROUPS = {
    "Temperature memory": ["tmax_anom_l0", "tmax_anom_l1", "tmax_anom_l2", "tmax_anom_3m",
                           "tmin_anom_l0", "tmin_anom_l1"],
    "Moisture memory": ["rain_anom_l0", "rain_anom_3m", "rain_anom_6m", "soil_anom_l0", "soil_anom_l1"],
    "Radiation": ["solar_anom_l0"],
    "Large-scale context": ["nat_tmax_anom_l0"],
    # Persistence strength depends on the season (EDA F7: lag-1 r = 0.39 for pre-monsoon targets,
    # 0.18 for monsoon). A linear model cannot learn that from separate features, so we provide the
    # products anomaly x sin/cos(target month): this lets Ridge/SVR fit a season-varying persistence.
    "Seasonal persistence": ["l0_x_sin", "l0_x_cos", "nat_x_sin", "nat_x_cos"],
    "Seasonality": ["month_sin", "month_cos", "target_tmax_clim"],
    "Geography": ["elevation_m", "lat", "lon"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def build_features(df: pd.DataFrame, keep_unlabelled: bool = False) -> pd.DataFrame:
    """
    df: processed monthly table from src.preprocessing. Returns one row per (district, target month).
    keep_unlabelled=True also keeps rows whose target month is not observed yet (the month after
    the last data month). These are real forecasts, used by the dashboard; their split is "forecast".
    """
    d = df.sort_values(["district_id", "date"]).copy()
    g = d.groupby("district_id")

    out = d[["district_id", "district", "province", "elevation_m", "lat", "lon"]].copy()
    out["input_date"] = d["date"]

    # 1. Temperature memory (lag 0 = month t, the latest known month)
    for lag in range(3):
        out[f"tmax_anom_l{lag}"] = g["tmax_c_anom"].shift(lag)
    out["tmax_anom_3m"] = out[["tmax_anom_l0", "tmax_anom_l1", "tmax_anom_l2"]].mean(axis=1)
    out["tmin_anom_l0"] = d["tmin_c_anom"]
    out["tmin_anom_l1"] = g["tmin_c_anom"].shift(1)

    # 2. Moisture memory - rolling windows end at month t (no future months)
    out["rain_anom_l0"] = d["rain_mm_anom"]
    out["rain_anom_3m"] = g["rain_mm_anom"].transform(lambda s: s.rolling(3).mean())
    out["rain_anom_6m"] = g["rain_mm_anom"].transform(lambda s: s.rolling(6).mean())
    out["soil_anom_l0"] = d["soil_moisture_anom"]
    out["soil_anom_l1"] = g["soil_moisture_anom"].shift(1)

    # 3. Radiation
    out["solar_anom_l0"] = d["solar_mj_anom"]

    # 4. National context at month t (mean over all districts in the same month)
    out["nat_tmax_anom_l0"] = d.groupby("date")["tmax_c_anom"].transform("mean")

    # Target: month t+1
    out["target_date"] = d["date"] + pd.DateOffset(months=1)
    out[TARGET] = g["tmax_c_anom"].shift(-1)
    out["target_tmax"] = g["tmax_c"].shift(-1)
    # Climatology of the target month: known in advance (training-period average), so it exists
    # even for a month that has not happened yet.
    clim = d.groupby(["district_id", "month"])["tmax_c_clim"].first()
    out["target_tmax_clim"] = clim.reindex(pd.MultiIndex.from_arrays(
        [out["district_id"], out["target_date"].dt.month])).to_numpy()
    out["target_season"] = pd.Categorical(out["target_date"].dt.month.map(P.SEASONS),
                                          categories=P.SEASON_ORDER, ordered=True)

    # 5. Seasonality of the TARGET month
    tm = out["target_date"].dt.month
    out["target_month"] = tm
    out["month_sin"] = np.sin(2 * np.pi * tm / 12)
    out["month_cos"] = np.cos(2 * np.pi * tm / 12)
    out["target_year"] = out["target_date"].dt.year

    # Seasonal persistence (interactions)
    out["l0_x_sin"] = out["tmax_anom_l0"] * out["month_sin"]
    out["l0_x_cos"] = out["tmax_anom_l0"] * out["month_cos"]
    out["nat_x_sin"] = out["nat_tmax_anom_l0"] * out["month_sin"]
    out["nat_x_cos"] = out["nat_tmax_anom_l0"] * out["month_cos"]

    # Rows without full history (first 5 months: 6-month window) or without a target (Dec 2025)
    n0 = len(out)
    out = out.dropna(subset=FEATURES if keep_unlabelled else FEATURES + [TARGET]).reset_index(drop=True)
    out.attrs["dropped_rows"] = n0 - len(out)
    out["target_year"] = out["target_year"].astype(int)
    out["target_month"] = out["target_month"].astype(int)
    out["split"] = assign_split(out["target_year"])
    out.loc[out[TARGET].isna(), "split"] = "forecast"
    return out


# ---------------------------------------------------------------------------
# Optional: large-scale ocean drivers (notebook 07)
# ---------------------------------------------------------------------------
# Indices from gee/extract_indices.py: anomalies vs a FIXED 1982-2010 base period,
# which ends before every validation/test year, so they are leakage-free in every fold.
OCEAN_GROUPS = {
    "ENSO (Nino-3.4)": ["nino34_l0", "nino34_3m"],
    "IOD (DMI)": ["dmi_l0", "dmi_3m"],
    "Tropical SST": ["trop_sst_l0", "trop_sst_3m"],
}
OCEAN_FEATURES = [f for g in OCEAN_GROUPS.values() for f in g]


def load_indices() -> pd.DataFrame:
    idx = pd.read_csv(C.DATA_RAW / "climate_indices.csv")
    idx["date"] = pd.to_datetime(dict(year=idx["year"], month=idx["month"], day=1))
    idx = idx.sort_values("date").set_index("date")
    for k in ["nino34", "dmi", "trop_sst"]:
        idx[f"{k}_l0"] = idx[k]
        idx[f"{k}_3m"] = idx[k].rolling(3).mean()      # 3-month mean ending at month t
    return idx[OCEAN_FEATURES]


def add_ocean_features(feat: pd.DataFrame, idx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach index values of the INPUT month t (known when forecasting month t+1)."""
    idx = load_indices() if idx is None else idx
    out = feat.copy()
    vals = idx.reindex(out["input_date"])
    for c in OCEAN_FEATURES:
        out[c] = vals[c].to_numpy()
    return out


def assign_split(year: pd.Series) -> pd.Series:
    return pd.Series(np.select([year <= C.TRAIN_YEARS[1], year <= C.VAL_YEARS[1]],
                               ["train", "validation"], "test"), index=year.index)


def extreme_threshold(feat: pd.DataFrame) -> float:
    """90th percentile of TRAINING target anomalies (never computed from val/test)."""
    return float(np.percentile(feat.loc[feat["split"] == "train", TARGET], C.EXTREME_HEAT_PERCENTILE))


def load_features() -> pd.DataFrame:
    return pd.read_parquet(C.DATA_PROCESSED / "features.parquet")
