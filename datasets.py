# ---------- datasets.py ----------
# Dataset definitions and metadata
from utils.ee_helpers import init_ee
import ee

# Initialize EE before any ImageCollection/Image calls
init_ee()

DICO = {
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
    },
    "modis_lai_fapar": {
        "dataset": ee.ImageCollection("MODIS/061/MCD15A3H").select(["Lai", "Fpar"]),
        "resolution": 500, "min": 0, "max": 10000,
    },
    "modis_lst": {
        "dataset": ee.ImageCollection("MODIS/061/MOD11A1").select(["LST_Day_1km", "LST_Night_1km"]),
        "resolution": 1000, "min": 7500, "max": 65535,
    },
    "modis_et": {
        "dataset": ee.ImageCollection("MODIS/061/MOD16A2").select(["ET"]),
        "resolution": 500, "min": 0, "max": 1000,
    },
    "chirps": {
        "dataset": ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation"),
        "resolution": 5566, "min": 0.0, "max": 200.0,
    },
    "era5": {
        "dataset": ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select("temperature_2m"),
        "resolution": 11132, "min": 180.0, "max": 330.0,
    },
    "terraclimate": {
        "dataset": ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE"),
        "resolution": 4638.3,
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
