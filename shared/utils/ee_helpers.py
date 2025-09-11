# Earth Engine init and download helpers
import ee, os
import logging
from retry import retry
from concurrent.futures import ThreadPoolExecutor
from config import EE_SERVICE_ACCOUNT, EE_KEY_PATH, TARGET_SCALE_METERS, DEFAULT_CRS

def init_ee():
    service_account = "signature-work@signature-work-403906.iam.gserviceaccount.com"
    json_key = "gee_key.json"
    ee.Initialize(
        ee.ServiceAccountCredentials(service_account, json_key),
        opt_url="https://earthengine-highvolume.googleapis.com"
    )

@retry(tries=5, delay=2, backoff=2)
def get_download_url(img, region, desc):
    return img.getDownloadUrl({
        "description": desc,
        "region": region,
        "crs": DEFAULT_CRS,
        "scale": TARGET_SCALE_METERS,
        "fileFormat": "GEO_TIFF"
    })

session = None

@retry(tries=3, delay=5, backoff=2)
def fetch_task(task):
    global session
    import requests
    if session is None:
        session = requests.Session()
    url, outp = task
    if os.path.exists(outp):
        return
    
    try:
        r = session.get(url, stream=True, timeout=(10, 900))
        r.raise_for_status()
        with open(outp, 'wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    except requests.exceptions.HTTPError as e:
        if "503" in str(e):
            print(f"⚠️  Google Earth Engine service temporarily unavailable (503). Retrying...")
            raise  # Let retry decorator handle it
        else:
            print(f"❌ HTTP Error downloading {outp}: {e}")
            raise
    except Exception as e:
        print(f"❌ Error downloading {outp}: {e}")
        raise