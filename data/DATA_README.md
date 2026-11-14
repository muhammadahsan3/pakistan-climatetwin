# Pakistan ClimateTwin: dataset

District-level monthly climate data for Pakistan's 126 districts, January 2000 to December 2025.
Extracted with Google Earth Engine by `gee/extract_data.py` and `gee/extract_indices.py`, then processed by notebooks 02 and 04.

## Files

| File | Rows | Description |
|---|---|---|
| `raw/climate_monthly_raw.csv` | 39,312 | One row per district and month: climate variables straight from GEE |
| `raw/climate_YYYY.csv` | 1,512 each | The same data, one file per year (as exported) |
| `raw/districts_static.csv` | 126 | District id, name, province, centroid lat/lon, area (km²), mean SRTM elevation (m) |
| `raw/districts.geojson` | 126 | District polygons (simplified, 500 m tolerance) |
| `raw/climate_indices.csv` | 528 | Monthly Niño-3.4, DMI (IOD) and tropical-mean SST anomalies, 1982–2025 (base 1982–2010) |
| `raw/sst_boxes_raw.csv` | 528 | Raw monthly mean SST of the four index boxes (°C) |
| `processed/climate_monthly.parquet` | 39,312 | Cleaned table + seasons + 2000–2019 climatology and anomalies |
| `processed/features.parquet` | 38,556 | Model-ready table: target (next-month Tmax anomaly), 23 features, train/validation/test split |
| `processed/district_trends.csv` | 126 | Mann-Kendall / Sen's slope / FDR results per district (Tmax, Tmin, Tmean, DTR, rainfall) |
| `processed/test_predictions_ocean.parquet` | 4,536 | Test-period (2023–2025) predictions of all models and baselines |

## Columns of `climate_monthly_raw.csv`

| Column | Unit | Meaning | Source |
|---|---|---|---|
| `district_id` | – | geoBoundaries v6.0.0 ADM2 `shapeID` | geoBoundaries |
| `year`, `month` | – | calendar month | – |
| `tmax_c` | °C | mean of daily maximum 2 m air temperature | ERA5-Land daily aggregates |
| `tmin_c` | °C | mean of daily minimum 2 m air temperature | ERA5-Land |
| `tmean_c` | °C | mean 2 m air temperature | ERA5-Land |
| `soil_moisture` | m³/m³ | volumetric soil water, 0–7 cm | ERA5-Land |
| `solar_mj` | MJ m⁻² day⁻¹ | net surface solar radiation | ERA5-Land |
| `era5_rain_mm` | mm/month | ERA5-Land total precipitation (cross-check only) | ERA5-Land |
| `rain_mm` | mm/month | monthly rainfall total | CHIRPS v3 daily |
| `era5_days`, `chirps_days` | – | number of daily images in the month | – |

## Sources and licences

- ERA5-Land: Copernicus Climate Change Service (C3S), Copernicus licence. GEE: `ECMWF/ERA5_LAND/DAILY_AGGR`
- CHIRPS v3: Climate Hazards Center, UC Santa Barbara. GEE: `UCSB-CHC/CHIRPS/V3/DAILY_RNL`
- NOAA OISST v2.1: NOAA NCEI, public domain. GEE: `NOAA/CDR/OISST/V2_1`
- SRTM: NASA/USGS, public domain. GEE: `USGS/SRTMGL1_003`
- geoBoundaries v6.0.0 (CC BY 4.0). GEE: `WM/geoLab/geoBoundaries/600/ADM2`

Cite the original datasets (see the references in `research/paper/main.tex`) when using this data.
