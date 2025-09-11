# ---------- datasets.py ----------
# Dataset definitions and metadata
import sys
import os

# Add parent directory to path to access shared modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Only import ee when actually needed
# import ee

# Note: EE initialization should be done explicitly when needed, not at import time

def get_datasets():
    """
    Get the DICO dictionary with EE objects. 
    This function should only be called after ee.Initialize() has been called.
    """
    import ee  # Import here to avoid hanging on module import
    return {
        "landsat": {
            "dataset": ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA"),
        "bands": {"RGB": ["B3", "B2", "B1"], "NIR": ["B4"], "panchromatic": ["B8"]},
        "resolution": 30, "min": 0.0, "max": 0.4,
    },
    "sentinel": {
        "dataset": ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED"),
        "bands": {"RGB": ["B4", "B3", "B2"], "NIR": ["B8"], "SWIR1": ["B11"], "SWIR2": ["B12"]},
        "resolution": 10, "min": 0.0, "max": 4500.0,
    },
    "modis_ndvi_evi": {
        "dataset": ee.ImageCollection("MODIS/061/MOD13Q1").select(["NDVI", "EVI"]),
        "resolution": 250, "min": -2000, "max": 10000,
        "temporal_frequency": "16-day",  # sub-monthly, use monthly means
    },
    "modis_lai_fapar": {
        "dataset": ee.ImageCollection("MODIS/061/MCD15A3H").select(["Lai", "Fpar"]),
        "resolution": 500, "min": 0, "max": 10000,
        "temporal_frequency": "4-day",  # sub-monthly, use monthly means
    },
    "modis_lst": {
        "dataset": ee.ImageCollection("MODIS/061/MOD11A1").select(["LST_Day_1km", "LST_Night_1km"]),
        "resolution": 1000, "min": 7500, "max": 65535,
        "temporal_frequency": "daily",  # daily, use monthly means
    },
    "modis_et": {
        "dataset": ee.ImageCollection("MODIS/061/MOD16A2").select(["ET"]),
        "resolution": 500, "min": 0, "max": 1000,
        "temporal_frequency": "8-day",  # sub-monthly, use monthly means
    },
    "chirps": {
        "dataset": ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation"),
        "resolution": 5566, "min": 0.0, "max": 200.0,
        "temporal_frequency": "daily",  # daily, use monthly means
    },
    "era5": {
        "dataset": ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select("temperature_2m"),
        "resolution": 11132, "min": 180.0, "max": 330.0,
        "temporal_frequency": "daily",  # daily, use monthly means
    },
    "terraclimate": {
        "dataset": ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE"),
        "resolution": 4638.3,
        "temporal_frequency": "monthly",  # already monthly, use monthly means
    },
    "soilgrids": {
        "layers": [
            "nitrogen_mean","sand_mean","silt_mean","clay_mean",
            "bdod_mean","cec_mean","cfvo_mean","ocd_mean","phh2o_mean"
        ],
        "crs": "EPSG:3857",
    },
    # Static (single-image) datasets:
    "srtm_slope": {
        "dataset": ee.Image("USGS/SRTMGL1_003"),
        "resolution": 30, "min": 0, "max": 60, "static": True,
    },
    "jrc_gsw": {
        "dataset": ee.Image("JRC/GSW1_4/GlobalSurfaceWater"),
        "resolution": 30, "min": 0, "max": 100, "static": True,
    },
    # Replaced worldcover with annual land-cover classification:
    "modis_mcd12q1": {
        "dataset": ee.ImageCollection("MODIS/061/MCD12Q1"),
        "resolution": 500,
        "bands": {"LC_Type1": ["LC_Type1"]},
        "min": 1, "max": 17, "static": True
    },
    # … keep adding as needed …
    }

# Initialize DICO as None - will be populated by calling get_datasets() after EE is initialized
DICO = None

def init_datasets():
    """Initialize the DICO dictionary with EE objects after EE is initialized."""
    global DICO
    if DICO is None:
        DICO = get_datasets()
