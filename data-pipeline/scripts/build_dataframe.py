# scripts/build_dataframe.py
import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import xarray as xr
except ImportError as e:
    raise ImportError("xarray is required to read NetCDF yield files. Please `pip install xarray netCDF4`.") from e

# ---- your config imports
from config.config import (
    START_YEAR, END_YEAR, FERTILIZER_DIR, YIELD_DIR,
    GEE_EXPORT_DIR, FINAL_PARQUET, RESAMPLING,
    TARGET_GRID_FROM_YIELD_DIR, IGNORE_PREFIXES
)
from config.datasets import DATASETS


# --------------------- logging helpers ---------------------
def _t(): return time.strftime("%Y-%m-%d %H:%M:%S")
def _log(msg: str, verbose: bool):
    if verbose:
        print(f"[{_t()}] {msg}", flush=True)


# --------------------- helpers: file listing ---------------------
def _list_tiffs(root: str) -> List[Path]:
    rootp = Path(root)
    files = []
    for p in rootp.rglob("*.tif*"):
        name = p.name
        if any(name.startswith(pref) for pref in IGNORE_PREFIXES):
            continue
        files.append(p)
    return files

def _list_netcdfs(root: str) -> List[Path]:
    rootp = Path(root)
    files = []
    for p in rootp.rglob("*.nc"):
        name = p.name
        if any(name.startswith(pref) for pref in IGNORE_PREFIXES):
            continue
        files.append(p)
    return files


# --------------------- target grid from NetCDF (yield) ---------------------
def _find_coord_names(ds: xr.Dataset) -> Tuple[str, str]:
    cand_lat = [n for n in list(ds.coords) + list(ds.variables) if n.lower() in ("lat", "latitude", "y")]
    cand_lon = [n for n in list(ds.coords) + list(ds.variables) if n.lower() in ("lon", "longitude", "x")]
    if not cand_lat or not cand_lon:
        raise ValueError("Could not identify lat/lon coordinate names in NetCDF.")
    lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else cand_lat[0])
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else cand_lon[0])
    return lat_name, lon_name

def _find_time_name(ds: xr.Dataset) -> Optional[str]:
    for name in list(ds.coords) + list(ds.variables):
        low = name.lower()
        if low in ("time", "year"):
            return name
    for d in ds.dims:
        if d.lower() in ("time", "year"):
            return d
    return None

def _choose_yield_var(ds: xr.Dataset, lat_name: str, lon_name: str) -> str:
    candidates = []
    for v in ds.data_vars:
        dims = set(map(str, ds[v].dims))
        if lat_name in dims and lon_name in dims:
            candidates.append(v)
    if not candidates:
        raise ValueError("No data variable in NetCDF has both latitude and longitude dims.")
    tname = _find_time_name(ds)
    if tname:
        for v in candidates:
            if tname in ds[v].dims:
                return v
    return candidates[0]

def _nc_to_target_profile_and_year_arrays(nc_path: Path, crop: str, verbose: bool) -> Tuple[dict, Dict[int, np.ndarray]]:
    """
    Read the crop yield NetCDF and return:
      - target_profile: (EPSG:4326) raster meta derived from the lon/lat vectors
      - year_to_arr:    dict[year] -> 2D array aligned so that
                        row 0, col 0 is at the NORTH-WEST corner (lat descending, lon increasing)
    This function is robust to var dims like (time, lat, lon), (lon, lat, time), (lat, lon), etc.
    """
    t0 = time.time()
    _log(f"[YIELD] Opening NetCDF for crop='{crop}': {nc_path}", verbose)
    ds = xr.open_dataset(nc_path)

    # ---- pick coord names ----
    lat_name = "lat" if "lat" in ds.coords else [c for c in ds.coords if c.lower() in ("lat","latitude","y")][0]
    lon_name = "lon" if "lon" in ds.coords else [c for c in ds.coords if c.lower() in ("lon","longitude","x")][0]
    tname = None
    for name in list(ds.coords) + list(ds.variables) + list(ds.dims):
        if str(name).lower() in ("time", "year"):
            tname = name
            break

    # ---- pick the yield variable (must have both lat & lon dims) ----
    var_name = None
    for v in ds.data_vars:
        dims = set(map(str, ds[v].dims))
        if lat_name in dims and lon_name in dims:
            var_name = v
            break
    if var_name is None:
        raise ValueError("No data variable in NetCDF has both latitude and longitude dims.")

    # ---- lon/lat vectors -> grid meta ----
    lats = np.asarray(ds[lat_name].values)
    lons = np.asarray(ds[lon_name].values)
    lon_increasing = True if lons.size < 2 else (lons[1] > lons[0])
    lat_increasing = True if lats.size < 2 else (lats[1] > lats[0])

    # mean step (robust to small irregularities)
    xres = float(np.abs(np.diff(lons)).mean()) if lons.size > 1 else 0.083333333333
    yres = float(np.abs(np.diff(lats)).mean()) if lats.size > 1 else 0.083333333333

    west  = float(lons.min()) - xres / 2.0
    north = float(lats.max()) + yres / 2.0

    target_profile = {
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": from_origin(west, north, xres, yres),
        "width": int(len(lons)),
        "height": int(len(lats)),
        "dtype": "float32",
        "nodata": np.nan,
    }
    _log(
        f"[YIELD] Grid {target_profile['width']}x{target_profile['height']} ; "
        f"lon_increasing={lon_increasing} lat_increasing={lat_increasing} ; "
        f"xres≈{xres:.5f}° yres≈{yres:.5f}°",
        verbose
    )

    # ---- force data to (lat, lon, [time]) then flip to (lat↓, lon↑) ----
    da = ds[var_name]
    dims = list(map(str, da.dims))
    order = [d for d in (lat_name, lon_name) if d in dims] + [d for d in dims if d not in (lat_name, lon_name)]
    da = da.transpose(*order)  # now lat, lon are the first 2 dims

    # Flip so row 0 = north, col 0 = west (matches from_origin)
    if lat_increasing:   # lat goes south->north in file → flip vertically
        da = da.sel({lat_name: ds[lat_name][::-1]})
    if not lon_increasing:  # lon goes east->west in file → flip horizontally
        da = da.sel({lon_name: ds[lon_name][::-1]})

    # ---- slice per year -> 2D arrays ----
    year_to_arr: Dict[int, np.ndarray] = {}
    if tname and (tname in da.dims):
        tvals = ds[tname].values
        try:
            years = pd.to_datetime(tvals).year.astype(int)
        except Exception:
            years = np.asarray(tvals).astype(int)

        years_unique = list(map(int, years.tolist()))
        _log(f"[YIELD] Years in file: {min(years_unique)}–{max(years_unique)} (n={len(years_unique)})", verbose)

        for i, yr in enumerate(years_unique):
            if not (START_YEAR <= yr <= END_YEAR):
                continue
            arr = np.asarray(da.isel({tname: i}).values, dtype="float32")
            # at this point arr is already (lat↓, lon↑)
            year_to_arr[int(yr)] = arr
    else:
        _log("[YIELD] No explicit time dimension; treating as static across selected years.", verbose)
        base = np.asarray(da.values, dtype="float32")
        for yr in range(START_YEAR, END_YEAR + 1):
            year_to_arr[int(yr)] = base.copy()

    ds.close()
    _log(f"[YIELD] Prepared per-year arrays in {time.time() - t0:.2f}s", verbose)
    return target_profile, year_to_arr


def _find_yield_nc_for_crop(yield_dir: str, crop: str) -> Path:
    crop_low = crop.lower()
    ncs = _list_netcdfs(yield_dir)
    for p in sorted(ncs):
        if crop_low in p.name.lower():
            return p
    if ncs:
        return sorted(ncs)[0]
    raise FileNotFoundError(f"No NetCDF files found under {yield_dir} for crop={crop}.")


# --------------------- raster reprojection ---------------------
def _resampling_enum(name: str):
    name = name.lower()
    if name == "average":  return Resampling.average
    if name == "bilinear": return Resampling.bilinear
    if name == "nearest":  return Resampling.nearest
    return Resampling.average

def _read_tif_reproject(src_path: Path, target_profile: dict, resampling: Resampling) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype("float32", copy=False)
        dst_arr = np.full((target_profile["height"], target_profile["width"]), np.nan, dtype="float32")
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            resampling=resampling,
            src_nodata=src.nodata if src.nodata is not None else np.nan,
            dst_nodata=np.nan,
        )
    return dst_arr

def _read_mask_reproject(mask_tif: Path, target_profile: dict) -> np.ndarray:
    with rasterio.open(mask_tif) as src:
        src_arr = src.read(1).astype("float32", copy=False)
        dst = np.full((target_profile["height"], target_profile["width"]), 0.0, dtype="float32")
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            resampling=Resampling.nearest,
            src_nodata=src.nodata if src.nodata is not None else 0.0,
            dst_nodata=0.0,
        )
    keep = (dst >= 0.5)
    return keep.reshape(-1)

# --------------------- NaN interpolation (3x3 neighbor mean) ---------------------
def _fill_nan_3x3(arr2d: np.ndarray, passes: int = 2) -> np.ndarray:
    """
    Fill NaNs by the mean of the 8-neighborhood (3x3 window), iterated 'passes' times.
    If a NaN has 0 valid neighbors, it stays NaN (no warning spam).
    """
    a = arr2d.astype(np.float32, copy=True)

    def _shift(src: np.ndarray, dy: int, dx: int) -> np.ndarray:
        H, W = src.shape
        out = np.full_like(src, np.nan, dtype=np.float32)
        y_src = slice(max(0, -dy), H - max(0, dy))
        x_src = slice(max(0, -dx), W - max(0, dx))
        y_dst = slice(max(0,  dy), H - max(0, -dy))
        x_dst = slice(max(0,  dx), W - max(0, -dx))
        out[y_dst, x_dst] = src[y_src, x_src]
        return out

    for _ in range(max(1, int(passes))):
        nanmask = np.isnan(a)
        if not nanmask.any():
            break

        neighs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighs.append(_shift(a, dy, dx))
        neighs = np.stack(neighs, axis=0)             # [8, H, W]
        valid  = ~np.isnan(neighs)                    # [8, H, W]
        cnt    = valid.sum(axis=0).astype(np.float32) # [H, W]
        neighs[~valid] = 0.0
        sm     = neighs.sum(axis=0)                   # [H, W]
        mean   = np.divide(sm, cnt, out=np.full_like(a, np.nan), where=cnt > 0)

        fillable = nanmask & (cnt > 0)
        a[fillable] = mean[fillable]

    return a

# --------------------- feature collection ---------------------
YEAR_REGEX = re.compile(r"(19|20)\d{2}")

def _sanitize_name(name: str) -> str:
    base = Path(name).stem
    base = YEAR_REGEX.sub("", base)
    base = re.sub(r"[^A-Za-z0-9]+", "_", base)
    return base.strip("_").lower()

def _group_yearly_rasters_by_dir(root: str, prefix: str, crop_filter: Optional[str], verbose: bool) -> Tuple[Dict[int, Dict[str, Path]], Dict[str, Path]]:
    per_year: Dict[int, Dict[str, Path]] = {}
    static: Dict[str, Path] = {}
    cf = crop_filter.lower() if crop_filter else None
    files = _list_tiffs(root)
    _log(f"[FERT] Scan {root} → {len(files)} tiffs", verbose)
    kept = 0
    for tif in files:
        name_san = _sanitize_name(tif.name)
        if cf and cf not in name_san:
            continue
        kept += 1
        year_m = YEAR_REGEX.search(tif.name)
        feat_name = f"{prefix}_{name_san}"
        if year_m:
            year = int(year_m.group(0))
            per_year.setdefault(year, {})
            per_year[year][feat_name] = tif
        else:
            static[feat_name] = tif
    _log(f"[FERT] Kept {kept} for crop='{crop_filter}' → yearly_years={len(per_year)} static={len(static)}", verbose)
    return per_year, static

def _collect_monthly_paths(monthly_root: str, start_year: int, end_year: int, verbose: bool) -> Dict[Tuple[int, int], Dict[str, Path]]:
    out: Dict[Tuple[int, int], Dict[str, Path]] = {}
    datasets = list(DATASETS.keys())
    total_hits = 0
    missing_by_ds = {k: 0 for k in datasets}
    for ds_key in datasets:
        for year in range(start_year, end_year + 1):
            year_dir = Path(monthly_root) / ds_key / f"{year:04d}"
            for month in range(1, 13):
                fpath = year_dir / f"{year:04d}{month:02d}.tif"
                if fpath.exists():
                    out.setdefault((year, month), {})
                    out[(year, month)][ds_key] = fpath
                    total_hits += 1
                else:
                    missing_by_ds[ds_key] += 1
    _log(f"[GEE] Monthly scan in {monthly_root}: {total_hits} files found across {len(datasets)} datasets", verbose)
    for ds_key, miss in missing_by_ds.items():
        _log(f"      - {ds_key}: missing {miss} of {(end_year-start_year+1)*12}", verbose)
    return out


# ---- dynamic conversions ----
def _convert_dynamic(ds_key: str, arr: np.ndarray) -> np.ndarray:
    if ds_key in ("modis_ndvi", "modis_evi"):
        return arr * 0.0001
    if ds_key in ("modis_lst_day", "modis_lst_night"):
        return arr * 0.02 - 273.15
    if ds_key == "era5_temp2m":
        return arr - 273.15
    return arr


# --------------------- main builder ---------------------
def build_dataframe(
    monthly_root: str,
    fertilizer_dir: str,
    yield_dir: str,
    out_parquet: str,
    start_year: int,
    end_year: int,
    crop: str,
    mask_tif: Optional[str] = None,
    verbose: bool = True
):
    run_t0 = time.time()
    crop = crop.lower().strip()
    if crop not in ("wheat", "maize"):
        raise ValueError("crop must be 'wheat' or 'maize'.")

    _log(f"=== BUILD START (crop={crop}, years={start_year}-{end_year}) ===", verbose)
    _log(f"[PATHS] monthly_root={monthly_root}", verbose)
    _log(f"[PATHS] fertilizer_dir={fertilizer_dir}", verbose)
    _log(f"[PATHS] yield_dir={yield_dir}", verbose)
    _log(f"[PATHS] out_parquet={out_parquet}", verbose)
    if mask_tif:
        _log(f"[PATHS] mask_tif={mask_tif}", verbose)

    # Validate mask if provided
    keep_mask = None
    if mask_tif:
        mp = Path(mask_tif)
        if not mp.exists():
            raise FileNotFoundError(f"Mask not found: {mp}")
    else:
        _log("[MASK] No mask provided; keeping all pixels.", verbose)

    # Yield grid / profile
    nc_path = _find_yield_nc_for_crop(yield_dir, crop)
    target_profile, yield_year_arrays = _nc_to_target_profile_and_year_arrays(nc_path, crop, verbose)

    height, width = target_profile["height"], target_profile["width"]
    rows = np.arange(height, dtype=np.int64); cols = np.arange(width, dtype=np.int64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    pixel_id_all = (rr * width + cc).reshape(-1).astype("int64")
    _log(f"[GRID] pixels total (full grid): {pixel_id_all.size}", verbose)

    # Land mask projection
    if mask_tif:
        keep_mask = _read_mask_reproject(Path(mask_tif), target_profile)
        kept = int(keep_mask.sum()); dropped = int((~keep_mask).sum())
        _log(f"[MASK] keep={kept} drop={dropped} (of {keep_mask.size})", verbose)
        if kept == 0:
            raise RuntimeError("Mask kept 0 pixels. Check mask scale/CRS or threshold.")

    ym_to_paths = _collect_monthly_paths(monthly_root, start_year, end_year, verbose)
    if not ym_to_paths:
        raise RuntimeError(f"No monthly rasters discovered under {monthly_root}. Ensure Step 1 completed and paths match DATASETS keys.")

    fert_yearly, fert_static = _group_yearly_rasters_by_dir(fertilizer_dir, prefix="fert", crop_filter=crop, verbose=verbose)

    resampling = _resampling_enum(RESAMPLING)

    # --- STATIC (other than fertilizer_* we treat as non-fert) ---
    static_arrays: Dict[str, np.ndarray] = {}
    _log(f"[STATIC] Reprojecting {len(fert_static)} static rasters...", verbose)
    t_static = time.time()
    for i, (feat_name, path) in enumerate(fert_static.items(), start=1):
        arr2d = _read_tif_reproject(path, target_profile, resampling)
        # Rule B (interpolate) applies to non-fert static; if this is 'fert_*', we won't fill
        if not feat_name.startswith("fert_"):
            arr2d = _fill_nan_3x3(arr2d, passes=2)
        arr = arr2d.reshape(-1)
        if keep_mask is not None: arr = arr[keep_mask]
        static_arrays[feat_name] = arr
        if i % 10 == 0 or i == len(fert_static):
            _log(f"  - static {i}/{len(fert_static)}: {feat_name} → shape={arr.shape}", verbose)
    _log(f"[STATIC] Done in {time.time() - t_static:.2f}s", verbose)

    writer = None; schema = None
    total_rows_written = 0; chunks_written = 0
    empty_months = 0

    pixel_id = pixel_id_all if keep_mask is None else pixel_id_all[keep_mask]
    ycol_name = f"yield_{crop}"

    for year in range(start_year, end_year + 1):
        _log(f"[YEAR] {year}", verbose)
        yearly_arrays: Dict[str, np.ndarray] = {}

        # Yield (NO interpolation; Rule A will drop rows where NaN)
        if year in yield_year_arrays:
            yarr2d = yield_year_arrays[year].astype("float32")
            yarr = yarr2d.reshape(-1)
            if keep_mask is not None: yarr = yarr[keep_mask]
            yearly_arrays[ycol_name] = yarr
            _log(f"  - yield present → {yarr.shape}", verbose)
        else:
            _log(f"  - yield missing for {year}; filling NaN", verbose)
            yearly_arrays[ycol_name] = np.full(pixel_id.shape, np.nan, dtype="float32")

        # Fertilizer yearly rasters (NO interpolation; Rule A will drop NaNs)
        f_year = fert_yearly.get(year, {})
        _log(f"  - fertilizer yearly rasters: {len(f_year)}", verbose)
        for feat_name, path in f_year.items():
            arr2d = _read_tif_reproject(path, target_profile, resampling)
            arr = arr2d.reshape(-1)
            if keep_mask is not None: arr = arr[keep_mask]
            yearly_arrays[feat_name] = arr

        for month in range(1, 13):
            paths = ym_to_paths.get((year, month), {})
            _log(f"    [MONTH {month:02d}] datasets present: {sorted(list(paths.keys()))}", verbose)
            if not paths:
                empty_months += 1
                continue

            t_chunk = time.time()
            dyn_cols: Dict[str, np.ndarray] = {}
            missing = []
            # Monthly dynamics: interpolate NaNs (Rule B) BEFORE flattening
            for ds_key in DATASETS.keys():
                fpath = paths.get(ds_key)
                if fpath is None:
                    missing.append(ds_key)
                    continue
                arr2d = _read_tif_reproject(fpath, target_profile, resampling)
                arr2d = _convert_dynamic(ds_key, arr2d)
                arr2d = _fill_nan_3x3(arr2d, passes=2)  # interpolate neighbors
                arr = arr2d.reshape(-1)
                if keep_mask is not None: arr = arr[keep_mask]
                dyn_cols[ds_key] = arr

            data = {
                "pixel_id": pixel_id,
                "year": np.full(pixel_id.shape, year, dtype="int32"),
                "month": np.full(pixel_id.shape, month, dtype="int16"),
            }
            for k, v in dyn_cols.items(): data[k] = v
            for k, v in yearly_arrays.items(): data[k] = v
            for k, v in static_arrays.items(): data[k] = v

            df = pd.DataFrame(data)

            # Replace obvious fill values → NaN
            df = df.replace([-9999, 1e20, np.inf, -np.inf], np.nan)

            # ----- Rule A: DROP rows if yield or ANY fertilizer feature is NaN
            fert_cols = [c for c in df.columns if c.startswith("fert_")]
            before_drop = len(df)
            cond_bad = df[ycol_name].isna()
            if fert_cols:
                cond_bad |= df[fert_cols].isna().any(axis=1)
            df = df[~cond_bad].copy()
            dropped_ruleA = before_drop - len(df)

            # (Optional) Count remaining NaNs (should be few after interpolation)
            remaining_nans = int(df.isna().sum().sum())

            rows_now = len(df)
            _log(
                f"      rows={rows_now:,} (dropped_ruleA={dropped_ruleA:,}; "
                f"missing_dyn={len(missing)}: {', '.join(missing) if missing else 'none'}; "
                f"remaining_nans_after_interp={remaining_nans})",
                verbose
            )

            if rows_now == 0:
                _log("      -> Skipping write (0 rows after Rule A)", verbose)
                continue

            # write
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                outp = Path(out_parquet)
                outp.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(out_parquet, schema, compression="snappy")
                _log(f"[WRITE] Opened ParquetWriter → {out_parquet}", verbose)
            else:
                # Align to initial schema (if column order fluctuates)
                table = table.select(schema.names)
                for i, field in enumerate(schema):
                    if not table.schema[i].equals(field):
                        import pyarrow.compute as pc
                        table = table.set_column(i, field.name, pc.cast(table.column(i), field.type))

            writer.write_table(table)
            chunks_written += 1
            total_rows_written += rows_now
            _log(f"      wrote rows={rows_now:,} | chunks_written={chunks_written} | total_rows={total_rows_written:,} | {time.time()-t_chunk:.2f}s", verbose)

    if writer is not None:
        writer.close()
        _log(f"=== BUILD DONE → {out_parquet} | chunks={chunks_written}, rows={total_rows_written:,} in {time.time()-run_t0:.2f}s ===", verbose)
    else:
        msg = [
            "No (year, month) chunk produced any rows — Parquet not created.",
            "Likely causes:",
            "  • Mask eliminated all pixels (keep=0) → check --mask_tif and threshold/CRS.",
            "  • No monthly rasters found → verify GEE export root and DATASETS keys.",
            "  • Rule A dropped all rows (yield/fert NaN) → inspect sources.",
            "  • Year range has no matching data in yield/fertilizer/monthly folders."
        ]
        raise RuntimeError("\n".join(msg))


def main():
    ap = argparse.ArgumentParser(description="Build monthly pixel-year-month dataframe with land-mask + NaN rules (drop for yield/fert, interpolate others).")
    ap.add_argument("--monthly_dir", type=str, default=GEE_EXPORT_DIR, help="Root of GEE monthly exports.")
    ap.add_argument("--fert_dir", type=str, default=FERTILIZER_DIR, help="Directory of fertilizer rasters.")
    ap.add_argument("--yield_dir", type=str, default=YIELD_DIR, help="Directory of yield NetCDFs.")
    ap.add_argument("--out", type=str, default=FINAL_PARQUET, help="Output Parquet file.")
    ap.add_argument("--start_year", type=int, default=START_YEAR)
    ap.add_argument("--end_year", type=int, default=END_YEAR)
    ap.add_argument("--crop", type=str, required=True, help="wheat or maize")
    ap.add_argument("--mask_tif", type=str, default="", help="Optional land mask GeoTIFF (1=land, 0=water)")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose progress logs")
    args = ap.parse_args()

    build_dataframe(
        monthly_root=args.monthly_dir,
        fertilizer_dir=args.fert_dir,
        yield_dir=args.yield_dir,
        out_parquet=args.out,
        start_year=args.start_year,
        end_year=args.end_year,
        crop=args.crop,
        mask_tif=(args.mask_tif or None),
        verbose=True if args.verbose else True  # default loud for debugging
    )

if __name__ == "__main__":
    main()
