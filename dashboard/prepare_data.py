"""
Prepare the small, pre-computed files the Streamlit dashboard reads.

Run once after notebooks 02-07 (from the project folder):
    python dashboard/prepare_data.py

Why pre-compute? The dashboard should open instantly and must not retrain models
or call Earth Engine while someone is clicking around. Everything heavy happens
here; the app only reads files in dashboard/data/.

What it writes
--------------
districts.geojson        district shapes (simplified)
districts.csv            static attributes + all Mann-Kendall / Sen trend results
monthly.parquet          district-month climate (2000-2025): Tmax, Tmin, rain, anomalies
clim_monthly.csv         district x calendar-month climatology (2000-2019)
test_predictions.parquet one-month-ahead test predictions 2023-2025, all models
metrics_*.csv            evaluation tables copied from reports/
forecast.csv             January 2026 forecast for every district (best model)
meta.json                dates, model names, headline numbers
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C, features as F, models as M, preprocessing as P  # noqa: E402

OUT = ROOT / "dashboard" / "data"
REP = ROOT / "reports"
BEST = "Ridge"          # best configuration: Ridge + ocean drivers (notebook 07)


def write_geojson_for_plotly(src: Path, dst: Path) -> None:
    """
    Plotly's geo maps (d3-geo) treat polygons on a sphere and expect the outer ring to run
    CLOCKWISE. Earth Engine writes GeoJSON with counter-clockwise outer rings (RFC 7946), which
    d3 interprets as "the whole globe minus the district". Re-orient every polygon.
    """
    import geopandas as gpd
    from shapely.geometry.polygon import orient
    from shapely.geometry import MultiPolygon, Polygon

    def cw(geom):
        if isinstance(geom, Polygon):
            return orient(geom, sign=-1.0)
        if isinstance(geom, MultiPolygon):
            return MultiPolygon([orient(p, sign=-1.0) for p in geom.geoms])
        return geom

    g = gpd.read_file(src)[["district_id", "district", "province", "geometry"]]
    g["geometry"] = g["geometry"].apply(cw)
    dst.write_text(g.to_json(drop_id=True), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    monthly = P.load_processed()

    # --- geometry + district attributes/trends -----------------------------------------
    write_geojson_for_plotly(C.DATA_RAW / "districts.geojson", OUT / "districts.geojson")
    trends = pd.read_csv(C.DATA_PROCESSED / "district_trends.csv")
    trends.to_csv(OUT / "districts.csv", index=False)

    # --- monthly climate ---------------------------------------------------------------
    keep = ["district_id", "date", "year", "month", "tmax_c", "tmin_c", "tmean_c", "rain_mm",
            "tmax_c_anom", "tmin_c_anom", "rain_mm_anom"]
    m = monthly[keep].copy()
    m.to_parquet(OUT / "monthly.parquet", index=False)
    clim = monthly[monthly["year"].between(*C.CLIMATOLOGY_YEARS)].groupby(["district_id", "month"]).agg(
        tmax_clim=("tmax_c", "mean"), tmin_clim=("tmin_c", "mean"), rain_clim=("rain_mm", "mean"),
        rain_p20=("rain_mm", lambda s: s.quantile(.2))).reset_index()
    clim.to_csv(OUT / "clim_monthly.csv", index=False)

    # --- test predictions + metrics ----------------------------------------------------
    pred = pd.read_parquet(C.DATA_PROCESSED / "test_predictions_ocean.parquet")
    pred.to_parquet(OUT / "test_predictions.parquet", index=False)
    for name in ["test_metrics_regression", "ocean_test_comparison", "test_skill_bootstrap_ci",
                 "test_metrics_extreme_heat", "lopo_results", "cv_comparison", "permutation_importance_groups"]:
        shutil.copy(REP / f"{name}.csv", OUT / f"metrics_{name}.csv")

    # --- January 2026 forecast ---------------------------------------------------------
    # Refit the best configuration on ALL labelled months (2000-2025) with the hyperparameters
    # selected in notebook 07, then predict the first unobserved month.
    params = json.load(open(C.MODELS_DIR / "best_params_ocean.json"))[BEST]
    feats = F.add_ocean_features(F.build_features(monthly, keep_unlabelled=True))
    cols = F.FEATURES + F.OCEAN_FEATURES
    labelled = feats[feats["split"] != "forecast"]
    fc = feats[feats["split"] == "forecast"].copy()
    model = M.fit_model(BEST, params, labelled, cols)
    fc["pred_anom"] = model.predict(fc[cols])
    phi = M.fit_damped_phi(labelled)
    fc["damped_anom"] = phi * fc["tmax_anom_l0"]

    # Uncertainty: empirical 10th-90th percentile of the test-period errors of the same model
    res = pred[F.TARGET] - pred[f"{BEST} + ocean"]
    lo, hi = np.percentile(res, [10, 90])
    fc["pred_tmax"] = fc["target_tmax_clim"] + fc["pred_anom"]
    fc["pred_tmax_lo"] = fc["pred_tmax"] + lo
    fc["pred_tmax_hi"] = fc["pred_tmax"] + hi
    thr_event = F.extreme_threshold(F.load_features())
    fc[["district_id", "district", "province", "input_date", "target_date", "tmax_anom_l0", "pred_anom",
        "damped_anom", "target_tmax_clim", "pred_tmax", "pred_tmax_lo", "pred_tmax_hi"]] \
        .to_csv(OUT / "forecast.csv", index=False)

    # --- meta ---------------------------------------------------------------------------
    reg = pd.read_csv(REP / "test_metrics_regression.csv", index_col=0)
    oc = pd.read_csv(REP / "ocean_test_comparison.csv", index_col=0)
    meta = {
        "data_start": str(monthly["date"].min().date()), "data_end": str(monthly["date"].max().date()),
        "n_districts": int(monthly["district_id"].nunique()),
        "forecast_month": str(fc["target_date"].iloc[0].date()),
        "best_model": f"{BEST} + ocean drivers",
        "best_test_rmse": float(oc.loc[BEST, "RMSE + ocean"]),
        "best_test_skill_vs_clim": float(oc.loc[BEST, "skill vs clim, + ocean"]),
        "clim_test_rmse": float(reg.loc["Climatology", "RMSE"]),
        "damped_test_rmse": float(reg.loc["Damped persistence", "RMSE"]),
        "interval_10_90": [float(lo), float(hi)],
        "phi": phi, "extreme_threshold_anom": thr_event,
        "climatology_years": list(C.CLIMATOLOGY_YEARS),
    }
    json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
    size = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"Wrote {len(list(OUT.iterdir()))} files ({size:.1f} MB) to {OUT}")
    print(f"Forecast for {meta['forecast_month']}: national mean predicted anomaly {fc['pred_anom'].mean():+.2f} °C")


if __name__ == "__main__":
    main()
