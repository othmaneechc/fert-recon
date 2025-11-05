# scripts/export_land_mask.py
import argparse, json, os, time, urllib.request
from pathlib import Path
import ee

# ---------- EE init ----------
def _init_ee():
    sa = os.environ.get("GEE_SERVICE_ACCOUNT")
    key = os.environ.get("GEE_PRIVATE_KEY")  # path OR raw JSON string
    if sa and key:
        p = os.path.expanduser(key)
        if os.path.exists(p):
            creds = ee.ServiceAccountCredentials(sa, key_file=p)
        else:
            key_json_str = key if isinstance(key, str) else json.dumps(key)
            creds = ee.ServiceAccountCredentials(sa, key_data=key_json_str)
        ee.Initialize(creds); return
    adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc and os.path.exists(os.path.expanduser(adc)):
        with open(adc, "r") as f: sa2 = json.load(f).get("client_email")
        creds = ee.ServiceAccountCredentials(sa2, key_file=adc)
        ee.Initialize(creds); return
    try:
        ee.Initialize(); return
    except Exception:
        pass
    raise RuntimeError(
        "No EE credentials. Set GEE_SERVICE_ACCOUNT + GEE_PRIVATE_KEY (path or JSON), "
        "or GOOGLE_APPLICATION_CREDENTIALS."
    )

def _country_aoi(country_name: str) -> ee.Geometry:
    fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", country_name))
    return ee.Feature(fc.first()).geometry()

def _download_url(img: ee.Image, region: ee.Geometry, scale_m: int, bandname: str):
    params = {
        "scale": scale_m,
        "crs": "EPSG:4326",
        "region": region.bounds(1e-3).getInfo()["coordinates"],
        "format": "GEO_TIFF",
        "bands": [bandname],
    }
    return img.rename(bandname).getDownloadURL(params)

def _save_url(url: str, out_path: Path, retries=5, sleep=5):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk: break
                    f.write(chunk)
            tmp.replace(out_path); return
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(sleep)

def export_land_mask(country: str, out_tif: str, threshold: int = 50, scale_m: int = 1000):
    """
    Land mask (1=land, 0=water) using JRC Global Surface Water occurrence.
    Water where occurrence >= threshold% at any time (1984-2019); land otherwise.
    """
    _init_ee()
    aoi = _country_aoi(country)
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")  # 0..100
    occ = gsw.unmask(0)  # missing -> 0
    land = occ.lt(threshold).rename("land_mask")  # True if NOT persistent water
    url = _download_url(land.toByte(), aoi, scale_m, bandname="land_mask")
    _save_url(url, Path(out_tif))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--out_tif", required=True, help="Output GeoTIFF path (land=1, water=0)")
    ap.add_argument("--threshold", type=int, default=50, help="JRC water occurrence threshold (0-100)")
    ap.add_argument("--scale_m", type=int, default=1000, help="Export scale in meters")
    args = ap.parse_args()
    export_land_mask(args.country, args.out_tif, args.threshold, args.scale_m)

if __name__ == "__main__":
    main()
