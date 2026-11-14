# Pakistan ClimateTwin

A machine-learning climate digital twin for Pakistan: district-level temperature trends,
warming hotspots, and next-month maximum-temperature anomaly prediction, built on
Google Earth Engine (ERA5-Land + CHIRPS).

- Full design: `PROJECT MASTER DOCUMENT.docx`
- Decisions since then, schedule and progress: [PLAN.md](PLAN.md)

## Project layout

```
data/raw/         Earth Engine exports (not committed to git)
data/processed/   cleaned, feature-engineered dataset
gee/              Earth Engine extraction scripts
notebooks/        02_preprocessing -> 03_eda -> 04_feature_engineering -> 05_model_training -> 06_model_evaluation -> 07_ocean_drivers
reports/          result tables (CV results, test metrics, bootstrap CIs, importance, LOPO)
src/              shared Python code (config.py holds every setting)
models/           trained models
dashboard/        Streamlit digital twin
research/         figures, literature review, IEEE paper
```

## Setup (each team member, once)

### 1. Python environment

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
python -m pip install -r requirements.txt
```

### 2. Google Earth Engine access

1. Sign up at https://code.earthengine.google.com/register with your Google account and choose
   **unpaid / academic (noncommercial) use**.
2. Create or select a **Google Cloud project** during registration and note its **project ID**
   (e.g. `climatetwin-123456`).
3. Put the ID in `src/config.py` (`EE_PROJECT = "climatetwin-123456"`) or pass `--project` each time.
4. The first run opens a browser window for Google sign-in. Sign in yourself; credentials are
   stored on your PC, not in the project.

### 3. Get the data (run from the project folder, with the venv active)

```bash
python gee/extract_data.py --check
```

Checks that every dataset and band exists, prints the date ranges, and counts the district polygons.

```bash
python gee/extract_data.py --years 2000
```

A one-year sample. Open `data/raw/climate_2000.csv` and sanity-check it: Tmax should be about
35–45 °C in June in Sindh and below 0 °C in winter in Gilgit-Baltistan, and rainfall should peak in July–August.

```bash
python gee/extract_data.py
```

The full 2000–2025 run.

```bash
python gee/extract_indices.py
```

Monthly ENSO (Niño-3.4), IOD (DMI) and tropical SST indices, 1982–2025, from NOAA OISST (used by notebook 07). It can be resumed: years already saved are skipped.

### 4. Run the analysis

Open the notebooks in order (`02` → `06`) and run all cells. Each notebook writes the inputs for the next one.
`05_model_training` takes about 20 minutes (hyperparameter search) and `06_model_evaluation` about 15 minutes (leave-one-province-out refits).

```bash
jupyter lab notebooks
```

## Boundary data (important)

Pakistan has about 160 **districts**. In the OCHA administrative hierarchy, ADM1 = province,
ADM2 = division and ADM3 = district. The geoBoundaries layer in the Earth Engine catalog stops at ADM2,
so it may contain about 30 divisions instead of districts. `--check` prints the count and warns you.

If it shows divisions:

1. Download the Pakistan administrative boundaries (ADM3 = districts) shapefile from
   HDX: https://data.humdata.org/dataset/cod-ab-pak
2. Upload it in the Earth Engine Code Editor: *Assets → New → Table upload (Shapefile)*.
3. In `src/config.py`, set `BOUNDARY_ASSET` to your asset path, `BOUNDARY_COUNTRY_FIELD = None`,
   and the ID/name fields to the shapefile's column names (e.g. `ADM3_PCODE`, `ADM3_EN`).

Record the source and version you use. It goes in the paper's dataset section.

## Variables extracted

| Column | Meaning | Source |
|--------|---------|--------|
| `tmax_c` | mean of daily maximum 2 m temperature (°C) | ERA5-Land daily |
| `tmin_c`, `tmean_c` | mean daily minimum / mean temperature (°C) | ERA5-Land daily |
| `soil_moisture` | volumetric soil water, 0–7 cm (m³/m³) | ERA5-Land daily |
| `solar_mj` | net surface solar radiation (MJ/m²/day) | ERA5-Land daily |
| `rain_mm` | monthly rainfall total (mm) | CHIRPS v3 daily |
| `era5_rain_mm` | ERA5-Land monthly precipitation (mm), cross-check only | ERA5-Land daily |
| `era5_days`, `chirps_days` | daily images per month (completeness check) | — |

Static, per district (`districts_static.csv`): province, centroid lat/lon, area, mean SRTM elevation.

## Data sources (cite in IEEE format in the paper)

- J. Muñoz-Sabater et al., "ERA5-Land: a state-of-the-art global reanalysis dataset for land applications," *Earth Syst. Sci. Data*, vol. 13, pp. 4349–4383, 2021.
- C. Funk et al., CHIRPS rainfall. **Check the correct citation for CHIRPS v3** on the CHC website; the 2015 *Scientific Data* paper describes v2.
- T. G. Farr et al., "The Shuttle Radar Topography Mission," *Rev. Geophys.*, vol. 45, 2007.
- B. Huang et al., "Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1," *J. Climate*, vol. 34, pp. 2923–2939, 2021 (OISST, used for the ENSO/IOD indices).
- N. H. Saji et al., "A dipole mode in the tropical Indian Ocean," *Nature*, vol. 401, pp. 360–363, 1999 (definition of the DMI).
- N. Gorelick et al., "Google Earth Engine: Planetary-scale geospatial analysis for everyone," *Remote Sens. Environ.*, vol. 202, pp. 18–27, 2017.
- Boundary dataset: whichever you choose (geoBoundaries or OCHA/HDX). Cite its version.

Verify every reference before it goes into the paper.
