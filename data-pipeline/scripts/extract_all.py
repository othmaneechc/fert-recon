#!/usr/bin/env python3
"""
extract_all.py  ·  2025-07-31
────────────────────────────────────────────────────────────────────────────
 1. Unzip every GEE export  → single-band tiles
 2. Crop + warp Cropland-maps & Yield-5-arcmin (2000-2015, 4 crops)
 3. Save all tiles in  output/processed/tiles/
 4. Build yearly multi-band stacks in output/processed/stack_yearly/
    (band description == tile filename stem)
"""
from __future__ import annotations
import re, zipfile, shutil, tempfile, warnings
from pathlib import Path
from typing import Tuple, Iterable, Dict

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds, transform
import xarray as xr
import numpy as np

# ──────────────────────────── CONSTANTS ───────────────────────────────
YEARS       = range(2000, 2016)
CROPS       = {"Maize", "Rice", "Soybean", "Wheat"}          # ← filter

ROOT        = Path("output")                      # project root
REGION      = ROOT / "region"                     # GEE yearly folders
CROP_DIR    = ROOT / "Cropland_Maps"              # *.tiff crop+fert+year
YIELD_DIR   = ROOT / "GlobalCropYield5min/GlobalCropYield5min1982_2015_V2.nc"        # NetCDFs per crop

TILES_DIR   = ROOT / "processed" / "tiles"
STACK_DIR   = ROOT / "processed" / "stack_yearly"
TILES_DIR.mkdir(parents=True, exist_ok=True)
STACK_DIR.mkdir(parents=True, exist_ok=True)

# Morocco bbox in lon/lat (EPSG:4326) – used for Cropland & Yield crops
MIN_LON, MAX_LON = -17.0, -1.0
MIN_LAT, MAX_LAT =  21.0, 36.0      # note: mins/lats for slicing Yield nc

# ──────────────────────────── UTILITIES ───────────────────────────────
def to_single_band(fp: Path) -> None:
    """Ensure a GeoTIFF is one-band; split multi-band files if needed."""
    with rasterio.open(fp) as src:
        if src.count == 1:
            return
        meta = src.meta.copy()
        for i in range(1, src.count + 1):
            meta.update(count=1)
            out_fp = fp.with_stem(f"{fp.stem}_b{i}")
            with rasterio.open(out_fp, "w", **meta) as dst:
                dst.write(src.read(i), 1)
        fp.unlink()                               # delete original

def unzip_tifs(zip_fp: Path, rename_fn) -> int:
    """Extract *.tif inside *zip_fp* → TILES_DIR  (rename via *rename_fn*)."""
    out = 0
    with zipfile.ZipFile(zip_fp) as zf, tempfile.TemporaryDirectory() as tmp:
        zf.extractall(tmp)
        for tif in Path(tmp).rglob("*.tif"):
            dst = TILES_DIR / rename_fn(tif)
            shutil.move(tif, dst)
            to_single_band(dst)
            out += 1
    return out

# ───────── reference grid (first tile found) ───────
def get_ref_grid() -> Tuple[Tuple[int,int], rasterio.Affine, rasterio.crs.CRS]:
    """Return (shape, transform, crs) of the 192×198 master grid."""
    try:
        fp = next(TILES_DIR.glob("*.tif"))
    except StopIteration:
        raise RuntimeError("Run GEE extraction first so a reference tile exists")
    with rasterio.open(fp) as src:
        return (src.height, src.width), src.transform, src.crs

def warp_to_ref(src_fp: Path, dst_fp: Path,
                ref_shape, ref_transform, ref_crs, nodata=0) -> None:
    """Re-project *src_fp* onto reference grid → *dst_fp* (skip if exists)."""
    if dst_fp.exists():
        return
    with rasterio.open(src_fp) as src:
        dtype = src.dtypes[0]
        dest_arr = np.zeros(ref_shape, dtype=dtype)
        reproject(
            source      = rasterio.band(src, 1),
            destination = dest_arr,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = ref_transform,
            dst_crs       = ref_crs,
            resampling    = Resampling.nearest,
            dst_nodata    = nodata,
        )
        meta = {
            "driver":"GTiff", "height":ref_shape[0], "width":ref_shape[1],
            "count":1, "dtype":dtype, "crs":ref_crs, "transform":ref_transform,
            "nodata":nodata
        }
        with rasterio.open(dst_fp, "w", **meta) as dst:
            dst.write(dest_arr, 1)

# ──────────── 1. GEE DATASET EXTRACTORS (unchanged) ──────────────────
def soilgrids():
    for z in (ROOT/"soilgrids").glob("*_mean.zip"):
        p = z.stem.replace("_mean","")
        unzip_tifs(z, lambda t: f"soilgrids_{p}_{t.stem.split('.')[-1]}.tif")

def simple_yearly(ds:str):
    """Extract datasets that still use yearly statistics."""
    for y in YEARS:
        for z in (REGION/ds/str(y)).glob(f"{ds}_*.zip"):
            parts = z.stem.split("_")
            var, stat = (parts[1], parts[2]) if len(parts)==3 else (parts[1],parts[2])
            unzip_tifs(z, lambda _p,v=var,s=stat,yy=y: f"{ds}_{v}_{s}_{yy}.tif")

def monthly_means(ds:str):
    """Extract datasets that now use monthly means (12 files per year)."""
    for y in YEARS:
        for z in (REGION/ds/str(y)).glob(f"{ds}_mean_m*.zip"):
            # Extract month from filename like "modis_ndvi_evi_mean_m01_2022.zip"
            parts = z.stem.split("_")
            # Find the month part (m01, m02, etc.)
            month_part = [p for p in parts if p.startswith('m') and len(p) == 3][0]
            month = month_part[1:]  # Extract "01" from "m01"
            
            def rename_func(p, ds_name=ds, month_num=month, year=y):
                # For multi-band datasets, preserve the band name
                base = p.stem.replace("download.", "")
                
                # Special handling for TerraClimate variables
                if ds_name == 'terraclimate':
                    terraclimate_vars = ["aet","def","pdsi","pet","pr","ro","soil",
                                       "srad","swe","tmmn","tmmx","vap","vpd","vs"]
                    # Check if base matches any terraclimate variable
                    for var in terraclimate_vars:
                        if var in base.lower():
                            return f"{ds_name}_{var}_mean_m{month_num}_{year}.tif"
                    # Fallback
                    return f"{ds_name}_{base.lower()}_mean_m{month_num}_{year}.tif"
                
                # Standard MODIS and other datasets
                elif base.lower() in ['ndvi', 'evi', 'lai', 'fpar', 'lst_day_1km', 'lst_night_1km', 'et']:
                    return f"{ds_name}_{base.lower()}_mean_m{month_num}_{year}.tif"
                else:
                    return f"{ds_name}_mean_m{month_num}_{year}.tif"
            
            unzip_tifs(z, rename_func)

def modis_combo(ds:str, clip_re:str):
    """Extract MODIS datasets - now using monthly means instead of yearly stats."""
    for y in YEARS:
        # Look for monthly mean files instead of yearly stats
        for z in (REGION/ds/str(y)).glob(f"{ds}_mean_m*.zip"):
            # Extract month from filename like "modis_ndvi_evi_mean_m01_2022.zip"
            parts = z.stem.split("_")
            month_part = [p for p in parts if p.startswith('m') and len(p) == 3][0]
            month = month_part[1:]  # Extract "01" from "m01"
            
            def rn(p, yy=y, month_num=month, ds_name=ds):
                base = re.sub(clip_re, "", p.stem.replace("download.", ""), flags=re.I)
                return f"{ds_name}_{base.lower()}_mean_m{month_num}_{yy}.tif"
            
            unzip_tifs(z, rn)

def modis_lc():
    base = REGION/"modis_mcd12q1"
    for y in YEARS:
        z = base/str(y)/f"modis_mcd12q1_{y}.zip"
        if z.exists():
            unzip_tifs(z, lambda p,yy=y: f"modis_mcd12q1_{p.stem}_{yy}.tif")

def jrc_gsw():
    for y in YEARS:
        for z in (REGION/"jrc_gsw"/str(y)).glob("*.zip"):
            unzip_tifs(z, lambda p,yy=y: f"{p.stem}_{yy}.tif")

def srtm():
    for y in YEARS:
        z = REGION/"srtm_slope"/str(y)/f"srtm_slope_{y}.zip"
        if z.exists():
            unzip_tifs(z, lambda _p,yy=y: f"srtm_slope_{yy}.tif")

# ──────────── 2. CROP THE ORIGINAL Cropland & Yield DATA ─────────────
def process_cropland(ref_shape, ref_transform, ref_crs):
    """Crop Cropland_Maps to Morocco bbox, warp to reference grid."""
    for tif in CROP_DIR.glob("*.tif*"):
        m = re.match(r"([A-Za-z]+)_([A-Za-z0-9]+)_(\d{4})\.tiff?", tif.name)
        if not m: continue
        crop, fert, year = m.groups()
        if crop not in CROPS or int(year) not in YEARS:
            continue
        with rasterio.open(tif) as src:
            win = from_bounds(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, src.transform)
            arr = src.read(1, window=win)
            if arr.size == 0:
                continue
            meta = src.meta.copy()
            meta.update({
                "height":win.height,"width":win.width,
                "transform": transform(win, src.transform)
            })
            tmp = TILES_DIR / f"__temp_{tif.stem}.tif"
            with rasterio.open(tmp,"w",**meta) as dst:
                dst.write(arr,1)
        out_fp = TILES_DIR / f"cropland_{crop.lower()}_{fert.lower()}_{year}.tif"
        warp_to_ref(tmp, out_fp, ref_shape, ref_transform, ref_crs)
        tmp.unlink(missing_ok=True)

def process_yield(ref_shape, ref_transform, ref_crs):
    """Split NetCDF Yield-5 arc-min, crop & warp each year."""
    for crop in CROPS:
        nc = YIELD_DIR / f"{crop}1982_2015.nc"
        if not nc.exists():
            warnings.warn(f"Yield file missing: {nc.name}")
            continue
        ds  = xr.open_dataset(nc, engine="netcdf4")
        var = list(ds.data_vars)[0]
        da  = ds[var]                              # DataArray
        da  = ( da.sel(time=slice("2000","2015"))
                  .sel(lon=slice(MIN_LON,MAX_LON),
                       lat=slice(MAX_LAT,MIN_LAT)) )
        da  = da.rename({"lon":"x","lat":"y"}).rio.write_crs("EPSG:4326")

        for yr in YEARS:
            arr = da.sel(time=str(yr)).squeeze("time").transpose("y","x").values
            tmp = TILES_DIR / f"__temp_yield_{crop}_{yr}.tif"
            with rasterio.open(
                tmp,"w",
                driver="GTiff",
                height=arr.shape[0], width=arr.shape[1],
                count=1, dtype=arr.dtype,
                crs="EPSG:4326",
                transform=da.rio.transform(),
            ) as dst:
                dst.write(arr,1)
            out_fp = TILES_DIR / f"yield_{crop.lower()}_{yr}.tif"
            warp_to_ref(tmp,out_fp,ref_shape,ref_transform,ref_crs)
            tmp.unlink(missing_ok=True)
        ds.close()

# ──────────── 3. BUILD YEARLY STACKS ──────────────────────────────────
def build_yearly_stacks():
    """Build yearly stacks including monthly mean data."""
    soil_tiles = sorted(TILES_DIR.glob("soilgrids_*.tif"))
    for yr in YEARS:
        # Include both monthly mean files and any remaining yearly files
        year_tiles = soil_tiles + sorted(TILES_DIR.glob(f"*_{yr}.tif")) + sorted(TILES_DIR.glob(f"*_m??_{yr}.tif"))
        if not year_tiles:
            continue
        with rasterio.open(year_tiles[0]) as src0:
            meta = src0.meta.copy()
        meta.update(count=len(year_tiles))
        out_fp = STACK_DIR / f"yearly_{yr}.tif"
        with rasterio.open(out_fp,"w",**meta) as dst:
            for i,fp in enumerate(year_tiles,1):
                with rasterio.open(fp) as src:
                    dst.write(src.read(1), i)
                    dst.set_band_description(i, fp.stem)
        print(f"📦  {yr}: stacked {len(year_tiles)} bands → {out_fp}")

# ──────────── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔄 1/4  GEE zip extraction → tiles …")
    soilgrids()
    
    # Datasets using monthly means (daily/sub-monthly frequency)
    print("📅 Processing monthly mean datasets...")
    monthly_means("chirps")
    monthly_means("era5") 
    monthly_means("terraclimate")
    modis_combo("modis_ndvi_evi",  r'_(mean)$')
    modis_combo("modis_lai_fapar", r'_(mean)$')
    modis_combo("modis_lst",       r'_(mean)$')
    monthly_means("modis_et")
    
    # Static datasets (unchanged)
    print("🏔️ Processing static datasets...")
    modis_lc()
    jrc_gsw()
    srtm()

    print("🔄 2/4  Establish reference grid …")
    ref_shape, ref_transform, ref_crs = get_ref_grid()

    print("🔄 3/4  Cropland-maps → tiles …")
    process_cropland(ref_shape, ref_transform, ref_crs)

    print("🔄 4/4  Yield-5′ → tiles …")
    process_yield(ref_shape, ref_transform, ref_crs)

    print("🔄 5/5  Build yearly stacks …")
    build_yearly_stacks()

    print("\n✅  Tiles directory  :", TILES_DIR)
    print("✅  Yearly stacks    :", STACK_DIR)
