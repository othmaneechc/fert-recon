# ---------- config.py ----------
# Global configuration and defaults
import os

TARGET_SCALE_METERS = 9265         # 5 arc-minutes ≈ 9265 meters
DEFAULT_CRS = "EPSG:3857"
EE_SERVICE_ACCOUNT = os.getenv(
    "EE_SERVICE_ACCOUNT",
    "signature-work@signature-work-403906.iam.gserviceaccount.com"
)
EE_KEY_PATH = os.getenv("EE_KEY_PATH", 
                        os.path.join(os.path.dirname(__file__), "gee_key.json"))
NATURAL_EARTH_GEOJSON = (
    "https://raw.githubusercontent.com/nvkelso/"
    "natural-earth-vector/master/geojson/"
    "ne_110m_admin_0_countries.geojson"
)

DEFAULT_COUNTRY = os.getenv("COUNTRY", "Morocco")
DEFAULT_START_YEAR = int(os.getenv("START_YEAR", "2022"))
DEFAULT_END_YEAR = int(os.getenv("END_YEAR", "2022"))
DEFAULT_OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "output")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))