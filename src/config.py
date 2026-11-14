"""
Central configuration for Pakistan ClimateTwin.

Every script and notebook imports settings from here, so the study period,
data splits and file paths are defined in exactly one place. Change a value
here and the whole pipeline follows.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL = ROOT / "data" / "external"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "research" / "figures"

# ---------------------------------------------------------------------------
# Study period
# ---------------------------------------------------------------------------
START_YEAR = 2000
END_YEAR = 2025

# Chronological split, by the *target* month (the month being predicted).
# Lag features are built from earlier months, so the first test target
# (Jan 2023) may legitimately use Dec 2022 data as input.
TRAIN_YEARS = (2000, 2019)
VAL_YEARS = (2020, 2022)
TEST_YEARS = (2023, 2025)

# The climatology used to compute anomalies must come from TRAINING years
# only. Using all years would leak information about the test period.
CLIMATOLOGY_YEARS = TRAIN_YEARS

# Threshold for the "extreme-heat month" classification task (PLAN.md, D3)
EXTREME_HEAT_PERCENTILE = 90

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Google Earth Engine
# ---------------------------------------------------------------------------
# Your Google Cloud project registered for Earth Engine. Either set it here,
# set the EE_PROJECT environment variable, or pass --project on the CLI.
EE_PROJECT = "pakistan-climatetwin"

# Climate datasets
ERA5_DAILY = "ECMWF/ERA5_LAND/DAILY_AGGR"      # daily aggregates -> monthly mean of daily max/min
CHIRPS_DAILY = "UCSB-CHC/CHIRPS/V3/DAILY_RNL"  # daily rainfall -> monthly totals
SRTM = "USGS/SRTMGL1_003"                      # elevation (static)

# District boundaries. See README "Boundary data" before changing.
# Option A (default): geoBoundaries in the GEE catalog, filtered to Pakistan.
# Option B: your own uploaded asset, e.g. "projects/<your-project>/assets/pak_adm3".
BOUNDARY_ASSET = "WM/geoLab/geoBoundaries/600/ADM2"
BOUNDARY_COUNTRY_FIELD = "shapeGroup"   # set to None for an asset that only contains Pakistan
BOUNDARY_COUNTRY_VALUE = "PAK"
BOUNDARY_ID_FIELD = "shapeID"           # unique, stable district identifier
BOUNDARY_NAME_FIELD = "shapeName"       # human-readable district name

# Provinces/territories (ADM1) - used to label each district with its
# province, which the leave-one-province-out validation needs.
PROVINCE_ASSET = "WM/geoLab/geoBoundaries/600/ADM1"
PROVINCE_NAME_FIELD = "shapeName"

# Spatial scale (metres) used when averaging pixels over each district.
# ERA5-Land is ~11 km and CHIRPS ~5.5 km. 5 km makes sure that even
# districts smaller than one ERA5 pixel still receive a value.
REDUCE_SCALE_M = 5000
