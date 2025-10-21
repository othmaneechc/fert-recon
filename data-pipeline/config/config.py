# config/config.py
import os
from pathlib import Path

# === Earth Engine auth ===
# If you have a service account, set env vars GEE_SERVICE_ACCOUNT and GEE_PRIVATE_KEY (path).
# Otherwise, we'll try user auth (interactive on first run).
GEE_SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT", None)
GEE_PRIVATE_KEY = os.environ.get("GEE_PRIVATE_KEY", None)  # path to JSON key or raw JSON string

# === AOI & time ===
# Default country for AOI (LSIB country name in EE). Change if needed.
DEFAULT_COUNTRY = "Morocco"

# Inclusive years
START_YEAR = 2002
END_YEAR   = 2019

# === Local data roots (your paths from the prompt) ===
FERTILIZER_DIR = "/data/oe23/fert-recon/data/Cropland_Maps"
YIELD_DIR      = "/data/oe23/fert-recon/data/GlobalCropYield5min"

# === Where to save things ===
# GEE monthly exports (GeoTIFFs)
GEE_EXPORT_DIR = "/data/oe23/fert-recon/exports/monthly"
Path(GEE_EXPORT_DIR).mkdir(parents=True, exist_ok=True)

# Final dataframe path (single Parquet file)
FINAL_PARQUET = "/data/oe23/fert-recon/datasets/fert_recommendation_monthly.parquet"
Path(Path(FINAL_PARQUET).parent).mkdir(parents=True, exist_ok=True)

# === Target grid definition ===
# We align ALL features to the grid of the FIRST TIFF found in YIELD_DIR.
# (This ensures consistent geotransform, crs, width/height, and stable pixel_id.)
TARGET_GRID_FROM_YIELD_DIR = YIELD_DIR

# Reprojection resampling strategy
RESAMPLING = "average"  # average for continuous variables

# NFS quirks: ignore '.nfs...' temp files silently
IGNORE_PREFIXES = (".nfs", "._")

# I/O safety
OVERWRITE_GEE_EXPORTS = False  # skip download if file exists
