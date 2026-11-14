"""
Preprocessing: turn the raw Earth Engine export into one clean analytical table.

Output: one row per district per month, with
  * the climate variables (already in physical units: deg C, mm, m3/m3, MJ/m2/day),
  * static district attributes (province, lat/lon, area, elevation),
  * a date column and a climatological season,
  * CLIMATOLOGY and ANOMALY columns for each climate variable.

KEY CONCEPT - climatology and anomaly
-------------------------------------
climatology(d, m) = average of variable for district d in calendar month m,
                    using the TRAINING years only (2000-2019).
anomaly(d, m, y)  = value(d, m, y) - climatology(d, m)

An anomaly removes the seasonal cycle and the fixed geographic differences
(Karachi is always warmer than Gilgit). What is left is "how unusual was this
month for this place" - the signal we want to analyse and predict.

Why training years only? If the climatology included 2023-2025, the anomaly of
every *training* row would contain information about the test period. That is
data leakage: test-set knowledge flowing into the training data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

CLIMATE_VARS = ["tmax_c", "tmin_c", "tmean_c", "rain_mm", "soil_moisture", "solar_mj"]

# Physically plausible ranges for Pakistan (generous bounds; used to flag errors,
# not to clip). Anything outside these is almost certainly a processing bug.
PLAUSIBLE = {
    "tmax_c": (-40, 55),
    "tmin_c": (-50, 40),
    "tmean_c": (-45, 45),
    "rain_mm": (0, 1500),
    "era5_rain_mm": (0, 1500),
    "soil_moisture": (0, 0.8),
    "solar_mj": (0, 40),
}

# Pakistan's climatological seasons (Pakistan Meteorological Department convention)
SEASONS = {12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
           3: "Pre-monsoon (MAM)", 4: "Pre-monsoon (MAM)", 5: "Pre-monsoon (MAM)",
           6: "Monsoon (JJAS)", 7: "Monsoon (JJAS)", 8: "Monsoon (JJAS)", 9: "Monsoon (JJAS)",
           10: "Post-monsoon (ON)", 11: "Post-monsoon (ON)"}
SEASON_ORDER = ["Winter (DJF)", "Pre-monsoon (MAM)", "Monsoon (JJAS)", "Post-monsoon (ON)"]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    climate = pd.read_csv(C.DATA_RAW / "climate_monthly_raw.csv")
    static = pd.read_csv(C.DATA_RAW / "districts_static.csv")
    return climate, static


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per check: what was checked and how many rows failed."""
    n_districts = df["district_id"].nunique()
    n_months = df.groupby(["year", "month"]).ngroups
    rows = [
        ("Rows", len(df), f"expected {n_districts} districts x {n_months} months = {n_districts * n_months}"),
        ("Missing values (any column)", int(df.isna().any(axis=1).sum()), ""),
        ("Duplicate district-month rows", int(df.duplicated(["district_id", "year", "month"]).sum()), ""),
        ("Months with < 28 daily ERA5 images", int((df["era5_days"] < 28).sum()), "incomplete month"),
        ("Months with < 28 daily CHIRPS images", int((df["chirps_days"] < 28).sum()), "incomplete month"),
    ]
    for col, (lo, hi) in PLAUSIBLE.items():
        if col in df:
            rows.append((f"{col} outside [{lo}, {hi}]", int(((df[col] < lo) | (df[col] > hi)).sum()), "implausible value"))
    return pd.DataFrame(rows, columns=["check", "count", "note"])


def fill_missing(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """
    Fill gaps by linear interpolation in time *within each district*.

    Climate variables are smooth from one month to the next, so interpolating
    between neighbours is far better than filling with a global mean (which
    would put a Gilgit value into Karachi). Gaps at the start/end of a series
    cannot be interpolated and are left for the caller to drop.
    """
    df = df.sort_values(["district_id", "year", "month"]).copy()
    before = int(df[cols].isna().sum().sum())
    df[cols] = df.groupby("district_id")[cols].transform(lambda s: s.interpolate(limit_area="inside"))
    return df, before - int(df[cols].isna().sum().sum())


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["season"] = pd.Categorical(df["month"].map(SEASONS), categories=SEASON_ORDER, ordered=True)
    # December belongs to the NEXT year's winter (Dec 2000 + Jan/Feb 2001 = winter 2001)
    df["season_year"] = df["year"] + (df["month"] == 12).astype(int)
    return df


def add_anomalies(df: pd.DataFrame, cols: list[str] = CLIMATE_VARS) -> pd.DataFrame:
    """Add <col>_clim and <col>_anom using a training-period climatology."""
    y0, y1 = C.CLIMATOLOGY_YEARS
    base = df[df["year"].between(y0, y1)]
    clim = base.groupby(["district_id", "month"])[cols].mean().add_suffix("_clim").reset_index()
    out = df.merge(clim, on=["district_id", "month"], how="left")
    for c in cols:
        out[f"{c}_anom"] = out[c] - out[f"{c}_clim"]
    return out


def build_dataset(save: bool = True) -> pd.DataFrame:
    climate, static = load_raw()
    df = climate.merge(static, on="district_id", how="left", validate="many_to_one")
    df, _ = fill_missing(df, CLIMATE_VARS)
    df = add_time_columns(df)
    df = add_anomalies(df)
    df = df.sort_values(["district_id", "date"]).reset_index(drop=True)
    if save:
        C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(C.DATA_PROCESSED / "climate_monthly.parquet", index=False)
    return df


def load_processed() -> pd.DataFrame:
    return pd.read_parquet(C.DATA_PROCESSED / "climate_monthly.parquet")


def annual_series(df: pd.DataFrame, var: str, season: str | None = None, how: str = "mean") -> pd.DataFrame:
    """
    District x year table of an annual (or seasonal) aggregate.
    Rainfall should use how="sum" (totals); temperatures use "mean".
    For winter (DJF) the season_year is used so a winter is not split across two years,
    and the first/last incomplete winters are dropped.
    """
    d = df if season is None else df[df["season"] == season]
    year_col = "season_year" if season == "Winter (DJF)" else "year"
    out = d.groupby(["district_id", year_col])[var].agg(how).unstack(year_col)
    if season == "Winter (DJF)":
        counts = d.groupby(["district_id", year_col])[var].count().unstack(year_col)
        out = out.loc[:, (counts == 3).all(axis=0)]
    return out
