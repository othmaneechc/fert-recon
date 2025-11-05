# scripts/export_gee_monthly.py
import argparse
import calendar
import json
import os
import time
from pathlib import Path
import urllib.request

import ee

from config.config import (
    GEE_SERVICE_ACCOUNT, GEE_PRIVATE_KEY,
    DEFAULT_COUNTRY, START_YEAR, END_YEAR,
    GEE_EXPORT_DIR, OVERWRITE_GEE_EXPORTS
)
from config.datasets import DATASETS

# ---------- EE init ----------
def _init_ee():
    """Initialize EE using a service account key or ADC, without gcloud."""
    sa = os.environ.get("GEE_SERVICE_ACCOUNT")
    key = os.environ.get("GEE_PRIVATE_KEY")  # path OR raw JSON string

    if sa and key:
        p = os.path.expanduser(key)
        if os.path.exists(p):
            creds = ee.ServiceAccountCredentials(sa, key_file=p)
        else:
            # ensure string for key_data
            key_json_str = key if isinstance(key, str) else json.dumps(key)
            creds = ee.ServiceAccountCredentials(sa, key_data=key_json_str)
        ee.Initialize(creds)
        return

    adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc and os.path.exists(os.path.expanduser(adc)):
        adc = os.path.expanduser(adc)
        try:
            with open(adc, "r") as f:
                sa2 = json.load(f).get("client_email")
        except Exception:
            sa2 = None
        if sa2:
            creds = ee.ServiceAccountCredentials(sa2, key_file=adc)
            ee.Initialize(creds)
            return

    try:
        ee.Initialize()
        return
    except Exception:
        pass

    raise RuntimeError(
        "No Earth Engine credentials found.\n"
        "Set either:\n"
        "  - GEE_SERVICE_ACCOUNT='svc@project.iam.gserviceaccount.com'\n"
        "  - GEE_PRIVATE_KEY='/path/to/key.json' (or the JSON string)\n"
        "or set GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'."
    )

# ---------- Helpers ----------
def _country_aoi(country_name: str) -> ee.Geometry:
    fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", country_name))
    return ee.Feature(fc.first()).geometry()

def _month_bounds(year: int, month: int):
    start = ee.Date.fromYMD(year, month, 1)
    ndays = calendar.monthrange(year, month)[1]
    end = start.advance(ndays, "day")
    return start, end

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
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=600) as resp, open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp_path.replace(out_path)
            return
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(sleep)

# ---------- Builders ----------
def _build_collection_monthly(ds_cfg: dict, start: ee.Date, end: ee.Date, aoi: ee.Geometry) -> ee.Image:
    ic = ee.ImageCollection(ds_cfg["ee_id"]).select(ds_cfg["band"]).filterDate(start, end)
    agg = ds_cfg.get("agg", "mean").lower()
    if agg == "sum":
        img = ic.sum()
    elif agg == "median":
        img = ic.median()
    else:
        img = ic.mean()
    return img.toFloat().clip(aoi)

def _build_vpd_monthly(ds_cfg: dict, start: ee.Date, end: ee.Date, aoi: ee.Geometry) -> ee.Image:
    src = ds_cfg["source"]
    ic = ee.ImageCollection(src).filterDate(start, end)
    t2m = ic.select("temperature_2m").mean()
    td2m = ic.select("dewpoint_temperature_2m").mean()

    t_c = t2m.subtract(273.15)
    td_c = td2m.subtract(273.15)

    def _svp(temp_c: ee.Image) -> ee.Image:
        # es (kPa) using Tetens formula
        return temp_c.multiply(17.27).divide(temp_c.add(237.3)).exp().multiply(0.6108)

    es = _svp(t_c)
    e = _svp(td_c)
    vpd = es.subtract(e)
    vpd = vpd.where(vpd.lt(0), 0)
    return vpd.toFloat().clip(aoi)

# ---------- Main export loop ----------
def export_country_monthly(country: str, start_year: int, end_year: int, out_root: str, selected=None):
    _init_ee()
    aoi = _country_aoi(country)

    for ds_key, ds in DATASETS.items():
        if selected and ds_key not in selected:
            continue

        dstype = ds["type"]
        scale_m = ds["scale_m"]
        # Safe logging even for derived datasets
        src_label = ds.get("ee_id", ds.get("source", "<derived>"))
        cadence = ds.get("cadence", "?")
        agg = ds.get("agg", "mean")
        print(f"\n[DATASET] {ds_key} → {src_label} ({cadence}, {agg})")

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                print(f"  - {ds_key} {year}-{month:02d}: preparing...", flush=True)
                start, end = _month_bounds(year, month)
                out_path = Path(out_root) / ds_key / f"{year:04d}" / f"{year:04d}{month:02d}.tif"
                if out_path.exists() and not OVERWRITE_GEE_EXPORTS:
                    print("    • exists, skipping")
                    continue

                try:
                    if dstype == "collection":
                        img = _build_collection_monthly(ds, start, end, aoi)
                    elif dstype == "derived_vpd":
                        img = _build_vpd_monthly(ds, start, end, aoi)
                    else:
                        print("    • unknown type, skipping")
                        continue

                    print("    • requesting URL...", flush=True)
                    url = _download_url(img, aoi, scale_m, bandname=ds_key)
                    print("    • downloading...", flush=True)
                    _save_url(url, out_path)
                    print(f"    • saved → {out_path}", flush=True)
                except Exception as e:
                    print(f"    ! failed: {e}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Export monthly GEE features over a country AOI.")
    parser.add_argument("--country", type=str, default=DEFAULT_COUNTRY)
    parser.add_argument("--start_year", type=int, default=START_YEAR)
    parser.add_argument("--end_year", type=int, default=END_YEAR)
    parser.add_argument("--out_dir", type=str, default=GEE_EXPORT_DIR)
    parser.add_argument("--datasets", type=str, default="", help="Comma-separated keys to export (subset).")
    args = parser.parse_args()

    selected = set([s.strip() for s in args.datasets.split(",") if s.strip()])
    export_country_monthly(
        country=args.country,
        start_year=args.start_year,
        end_year=args.end_year,
        out_root=args.out_dir,
        selected=selected if selected else None
    )

if __name__ == "__main__":
    main()
