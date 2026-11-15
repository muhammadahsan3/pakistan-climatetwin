"""
Extract district-level monthly climate data for Pakistan from Google Earth Engine.

WHAT THIS DOES
--------------
For every district and every month from START_YEAR to END_YEAR it computes:

    tmax_c          mean of the DAILY maximum 2 m air temperature   (deg C)   <- main variable
    tmin_c          mean of the daily minimum 2 m air temperature   (deg C)
    tmean_c         mean 2 m air temperature                        (deg C)
    soil_moisture   volumetric soil water, top layer 0-7 cm         (m3/m3)
    solar_mj        net surface solar radiation, daily mean         (MJ/m2/day)
    era5_rain_mm    ERA5-Land monthly precipitation total           (mm)  - cross-check only
    rain_mm         CHIRPS monthly rainfall total                   (mm)  - main rainfall variable
    era5_days, chirps_days   number of daily images that went into the month (completeness check)

and, once, static district attributes: province, centroid lat/lon, area, mean elevation.

WHY MEAN OF DAILY MAXIMA?
-------------------------
"Maximum temperature" for a month can mean (a) the single hottest hour in the month, or
(b) the average of each day's maximum. Climate studies normally use (b): it describes
typical daytime heat, while (a) is one extreme value and is very noisy. We therefore
build months from ERA5-Land *daily* aggregates instead of the monthly product.

HOW THE SPATIAL AVERAGING WORKS
-------------------------------
`reduceRegions` takes every pixel whose centre falls inside a district polygon and
averages it. At REDUCE_SCALE_M = 5 km, Earth Engine resamples the ~11 km ERA5 grid
so even small urban districts get pixels.

USAGE (run from the project folder)
-----------------------------------
    python gee/extract_data.py --check            # 1. verify datasets, bands and boundaries
    python gee/extract_data.py --years 2000       # 2. sample: one year only, then inspect the CSV
    python gee/extract_data.py                    # 3. full run (resumable; skips finished years)

Add --project YOUR_CLOUD_PROJECT if EE_PROJECT is not set in src/config.py or the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import ee
import pandas as pd

# Make `src` importable when this file is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

KELVIN = 273.15

# ERA5-Land DAILY_AGGR bands we need, and what we turn each into.
ERA5_BANDS = {
    "temperature_2m_max": "tmax_c",
    "temperature_2m_min": "tmin_c",
    "temperature_2m": "tmean_c",
    "volumetric_soil_water_layer_1": "soil_moisture",
    "surface_net_solar_radiation_sum": "solar_mj",
    "total_precipitation_sum": "era5_rain_mm",
}

OUT_COLUMNS = ["district_id", "year", "month", *ERA5_BANDS.values(),
               "rain_mm", "era5_days", "chirps_days"]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def init_ee(project: str | None) -> None:
    project = project or C.EE_PROJECT or os.environ.get("EE_PROJECT")
    if not project:
        sys.exit("No Earth Engine project given. Use --project, set EE_PROJECT, "
                 "or edit EE_PROJECT in src/config.py.")
    # Automated runs (GitHub Actions): a service-account key passed as an environment variable.
    # The key itself is never stored in the repository.
    key_json = os.environ.get("EE_SERVICE_ACCOUNT_KEY")
    if key_json:
        info = json.loads(key_json)
        credentials = ee.ServiceAccountCredentials(info["client_email"], key_data=key_json)
        ee.Initialize(credentials, project=project)
        print(f"Earth Engine initialised with a service account (project: {project})")
        return

    creds = Path.home() / ".config" / "earthengine" / "credentials"
    if not creds.exists():
        print("Not authenticated yet - opening the Google sign-in flow ...")
        ee.Authenticate(auth_mode="localhost")
    try:
        ee.Initialize(project=project)
    except ee.EEException as e:
        msg = str(e)
        if "has not been used" in msg or "SERVICE_DISABLED" in msg:
            sys.exit(f"The Earth Engine API is not enabled for project '{project}'.\n"
                     "Enable it here, wait 2-5 minutes, then re-run:\n"
                     f"  https://console.cloud.google.com/apis/library/earthengine.googleapis.com?project={project}")
        if "not registered" in msg.lower() or "permission" in msg.lower():
            sys.exit(f"Project '{project}' is not registered for Earth Engine (or your account lacks access).\n"
                     "Register it for noncommercial use: https://code.earthengine.google.com/register\n"
                     f"Details: {msg}")
        raise
    print(f"Earth Engine initialised (project: {project})")


def load_districts() -> ee.FeatureCollection:
    fc = ee.FeatureCollection(C.BOUNDARY_ASSET)
    if C.BOUNDARY_COUNTRY_FIELD:
        fc = fc.filter(ee.Filter.eq(C.BOUNDARY_COUNTRY_FIELD, C.BOUNDARY_COUNTRY_VALUE))
    return fc


def load_provinces() -> ee.FeatureCollection:
    return (ee.FeatureCollection(C.PROVINCE_ASSET)
            .filter(ee.Filter.eq("shapeGroup", C.BOUNDARY_COUNTRY_VALUE)))


def detect_chirps_band() -> str:
    bands = ee.ImageCollection(C.CHIRPS_DAILY).first().bandNames().getInfo()
    return "precipitation" if "precipitation" in bands else bands[0]


# ---------------------------------------------------------------------------
# --check : verify everything before spending time on a full extraction
# ---------------------------------------------------------------------------
def date_range(coll: ee.ImageCollection) -> tuple[str, str]:
    r = coll.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
    lo, hi = r.get("min"), r.get("max")
    return (ee.Date(lo).format("YYYY-MM-dd").getInfo(),
            ee.Date(hi).format("YYYY-MM-dd").getInfo())


def run_check() -> None:
    ok = True
    print("\n=== ERA5-Land daily aggregates ===")
    era = ee.ImageCollection(C.ERA5_DAILY)
    era_bands = era.first().bandNames().getInfo()
    print("Date range:", *date_range(era))
    for b in ERA5_BANDS:
        found = b in era_bands
        ok &= found
        print(f"  {'OK ' if found else 'MISSING'} {b}")

    print("\n=== CHIRPS daily rainfall ===")
    chirps = ee.ImageCollection(C.CHIRPS_DAILY)
    print("Bands:", chirps.first().bandNames().getInfo(), "-> using", detect_chirps_band())
    print("Date range:", *date_range(chirps))

    print("\n=== District boundaries ===")
    d = load_districts()
    n = d.size().getInfo()
    print(f"Asset: {C.BOUNDARY_ASSET}\nNumber of units: {n}")
    names = d.aggregate_array(C.BOUNDARY_NAME_FIELD).getInfo()
    print("First 15:", sorted(names)[:15])
    if n < 100:
        print("WARNING: Pakistan has ~160 districts. Fewer than 100 units suggests this layer "
              "is DIVISIONS, not districts. See README 'Boundary data'.")
    p = load_provinces()
    print(f"Provinces/territories: {p.size().getInfo()} ->",
          sorted(p.aggregate_array(C.PROVINCE_NAME_FIELD).getInfo()))

    print("\nAll required ERA5 bands present." if ok else
          "\nSome ERA5 bands are missing - update ERA5_BANDS in this script.")


# ---------------------------------------------------------------------------
# Static district attributes (run once)
# ---------------------------------------------------------------------------
def extract_static(districts: ee.FeatureCollection) -> None:
    out_csv = C.DATA_RAW / "districts_static.csv"
    out_geo = C.DATA_RAW / "districts.geojson"
    if out_csv.exists() and out_geo.exists():
        print("Static district data already present - skipping.")
        return

    provinces = load_provinces()

    def add_attrs(f):
        f = ee.Feature(f)
        centroid = f.geometry().centroid(maxError=100)
        prov = provinces.filterBounds(centroid).first()
        coords = centroid.coordinates()
        return f.set({
            "district_id": f.get(C.BOUNDARY_ID_FIELD),
            "district": f.get(C.BOUNDARY_NAME_FIELD),
            "province": ee.Algorithms.If(prov, ee.Feature(prov).get(C.PROVINCE_NAME_FIELD), "Unknown"),
            "lon": coords.get(0),
            "lat": coords.get(1),
            "area_km2": f.geometry().area(maxError=100).divide(1e6),
        })

    attrs = districts.map(add_attrs)
    elev = ee.Image(C.SRTM).select("elevation").rename("elevation_m")
    # 1 km is plenty for a district-average elevation and much faster than 30 m.
    attrs = elev.reduceRegions(attrs, ee.Reducer.mean().setOutputs(["elevation_m"]),
                               scale=1000, tileScale=4)

    keep = ["district_id", "district", "province", "lat", "lon", "area_km2", "elevation_m"]
    # Simplify polygons (500 m tolerance) so the GeoJSON stays small for the dashboard.
    simplified = attrs.map(lambda f: ee.Feature(ee.Feature(f).geometry().simplify(500))
                           .copyProperties(f, keep))
    gj = call_with_retry(lambda: simplified.getInfo(), "static attributes")

    rows = [feat["properties"] for feat in gj["features"]]
    pd.DataFrame(rows)[keep].sort_values("district_id").to_csv(out_csv, index=False)
    with open(out_geo, "w", encoding="utf-8") as fh:
        json.dump(gj, fh)
    print(f"Saved {len(rows)} districts -> {out_csv.name}, {out_geo.name}")


# ---------------------------------------------------------------------------
# Monthly climate
# ---------------------------------------------------------------------------
def monthly_image(year: int, month: int, rain_band: str) -> ee.Image:
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    era = ee.ImageCollection(C.ERA5_DAILY).filterDate(start, end)
    chirps = ee.ImageCollection(C.CHIRPS_DAILY).filterDate(start, end).select(rain_band)

    tmax = era.select("temperature_2m_max").mean().subtract(KELVIN).rename("tmax_c")
    tmin = era.select("temperature_2m_min").mean().subtract(KELVIN).rename("tmin_c")
    tmean = era.select("temperature_2m").mean().subtract(KELVIN).rename("tmean_c")
    soil = era.select("volumetric_soil_water_layer_1").mean().rename("soil_moisture")
    solar = era.select("surface_net_solar_radiation_sum").mean().divide(1e6).rename("solar_mj")  # J -> MJ
    era_rain = era.select("total_precipitation_sum").sum().multiply(1000).rename("era5_rain_mm")  # m -> mm
    rain = chirps.sum().rename("rain_mm")

    return (ee.Image.cat([tmax, tmin, tmean, soil, solar, era_rain, rain])
            .set({"year": year, "month": month,
                  "era5_days": era.size(), "chirps_days": chirps.size()}))


def tag_with(img: ee.Image):
    """Return a function that drops the geometry and stamps year/month on each feature."""
    def _tag(f):
        return ee.Feature(None, ee.Feature(f).toDictionary()).set({
            "year": img.get("year"), "month": img.get("month"),
            "era5_days": img.get("era5_days"), "chirps_days": img.get("chirps_days"),
        })
    return _tag


def extract_year(districts: ee.FeatureCollection, year: int, rain_band: str) -> pd.DataFrame:
    slim = districts.map(lambda f: ee.Feature(f.geometry(), {"district_id": f.get(C.BOUNDARY_ID_FIELD)}))

    per_month = []
    for month in range(1, 13):
        img = monthly_image(year, month, rain_band)
        reduced = img.reduceRegions(slim, ee.Reducer.mean(), scale=C.REDUCE_SCALE_M, tileScale=4)
        per_month.append(reduced.map(tag_with(img)))

    fc = ee.FeatureCollection(per_month).flatten()
    data = call_with_retry(lambda: fc.getInfo(), f"year {year}")
    df = pd.DataFrame([f["properties"] for f in data["features"]])
    return df.reindex(columns=OUT_COLUMNS).sort_values(["district_id", "month"])


def call_with_retry(fn, label: str, tries: int = 4):
    """Earth Engine sometimes times out or rate-limits; wait and try again."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except ee.EEException as e:
            if attempt == tries:
                raise
            wait = 20 * attempt
            print(f"  {label}: Earth Engine error ({e}); retrying in {wait}s ...")
            time.sleep(wait)


def parse_years(s: str | None) -> list[int]:
    if not s:
        return list(range(C.START_YEAR, C.END_YEAR + 1))
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in s.split(",")]


def combine_years() -> None:
    files = sorted(C.DATA_RAW.glob("climate_*.csv"))
    expected = C.END_YEAR - C.START_YEAR + 1
    if len(files) < expected:
        print(f"{len(files)}/{expected} years extracted - combined file not written yet.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = C.DATA_RAW / "climate_monthly_raw.csv"
    df.to_csv(out, index=False)
    print(f"Combined {len(files)} years, {len(df):,} rows -> {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--check", action="store_true", help="verify datasets and boundaries, then exit")
    ap.add_argument("--years", help="e.g. 2000  |  2000-2005  |  2000,2010")
    ap.add_argument("--force", action="store_true", help="re-extract years that already have a CSV")
    args = ap.parse_args()

    init_ee(args.project)
    if args.check:
        run_check()
        return

    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    districts = load_districts()
    rain_band = detect_chirps_band()
    extract_static(districts)

    for year in parse_years(args.years):
        out = C.DATA_RAW / f"climate_{year}.csv"
        if out.exists() and not args.force:
            print(f"{year}: already extracted - skipping")
            continue
        t0 = time.time()
        df = extract_year(districts, year, rain_band)
        df.to_csv(out, index=False)
        print(f"{year}: {len(df):,} rows in {time.time() - t0:.0f}s -> {out.name}")

    combine_years()


if __name__ == "__main__":
    main()
