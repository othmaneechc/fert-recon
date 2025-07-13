import os
import ee
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import TARGET_SCALE_METERS
from datasets import DICO
from utils.geometry import country_bbox
from utils.ee_helpers import init_ee, get_download_url, fetch_task

# Datasets that use the generic multi‐band composite workflow
SPLIT_DS = [
    'modis_ndvi_evi', 'modis_lai_fapar', 'modis_lst', 
    'modis_et', 'sentinel', 'landsat'
]
# Specialized single‐collection workflows
OTHER_DS = ['chirps', 'era5', 'terraclimate']


def run_region(dataset: str, start_date: str, end_date: str, output_dir: str, country: str):
    # Initialize Earth Engine
    init_ee()

    # 1) Country bounding box in EPSG:4326
    bbox = country_bbox(country)
    geom = ee.Geometry.Rectangle(bbox)

    # 2) Handle static datasets (Image or ImageCollection)
    if DICO[dataset].get('static', False):
        ds_obj = DICO[dataset]['dataset']
        # If it's an ImageCollection, filter by year
        if hasattr(ds_obj, 'filterDate'):
            coll = ds_obj.filterDate(start_date, end_date).filterBounds(geom)
            if coll.size().getInfo() == 0:
                print(f"No static images for {dataset} in {start_date[:4]}, skipping.")
                return
            img = coll.first()
        else:
            img = ds_obj
        # Clip and download
        img_clipped = img.clip(geom)
        year = start_date[:4]
        desc = f"{dataset}_{year}"
        out_zip = os.path.join(output_dir, f"{desc}.zip")
        url = get_download_url(img_clipped, bbox, desc)
        fetch_task((url, out_zip))
        print(f"✅ Static {dataset} for {year} → {out_zip}")
        return

    # 3) Filter the ImageCollection for non-static datasets
    coll = DICO[dataset]['dataset'] \
               .filterDate(start_date, end_date) \
               .filterBounds(geom)
    if coll.size().getInfo() == 0:
        print(f"No data for {dataset} in {start_date[:4]}, skipping.")
        return

    year = start_date[:4]
    metrics = {}

    # 4a) Generic split‐dataset: mean/min/max/stdDev/p25/p50/p75
    if dataset in SPLIT_DS:
        metrics = {
            'mean': coll.mean(),
            'min': coll.min(),
            'max': coll.max(),
            'stdDev': coll.reduce(ee.Reducer.stdDev()),
            'p25': coll.reduce(ee.Reducer.percentile([25])),
            'p50': coll.reduce(ee.Reducer.percentile([50])),
            'p75': coll.reduce(ee.Reducer.percentile([75])),
        }

    # 4b) CHIRPS precipitation stats
    elif dataset == 'chirps':
        summary = ee.Image.cat([
            coll.sum().rename('precip_sum'),
            coll.mean().rename('precip_mean'),
            coll.max().rename('precip_max'),
            coll.reduce(ee.Reducer.stdDev()).rename('precip_stdDev'),
            coll.reduce(ee.Reducer.percentile([25,50,75]))
                .rename(['precip_p25','precip_p50','precip_p75'])
        ])
        metrics = {bn: summary.select(bn) for bn in summary.bandNames().getInfo()}

    # 4c) ERA5 temperature stats
    elif dataset == 'era5':
        summary = ee.Image.cat([
            coll.mean().rename('temp_mean'),
            coll.max().rename('temp_max'),
            coll.reduce(ee.Reducer.stdDev()).rename('temp_stdDev'),
            coll.reduce(ee.Reducer.percentile([25,50,75]))
                .rename(['temp_p25','temp_p50','temp_p75'])
        ])
        metrics = {bn: summary.select(bn) for bn in summary.bandNames().getInfo()}

    # 4d) TerraClimate per-variable stats
    elif dataset == 'terraclimate':
        vars = [
            "aet","def","pdsi","pet","pr","ro","soil",
            "srad","swe","tmmn","tmmx","vap","vpd","vs"
        ]
        stats = []
        for v in vars:
            band = coll.select(v)
            stats.extend([
                band.mean().rename(f"{v}_mean"),
                band.min().rename(f"{v}_min"),
                band.max().rename(f"{v}_max"),
                band.reduce(ee.Reducer.stdDev()).rename(f"{v}_stdDev"),
                band.reduce(ee.Reducer.percentile([25,50,75]))
                    .rename([f"{v}_p25", f"{v}_p50", f"{v}_p75"])
            ])
        summary = ee.Image.cat(stats)
        metrics = {bn: summary.select(bn) for bn in summary.bandNames().getInfo()}

    else:
        print(f"No processing logic for {dataset}, skipping.")
        return

    # 5) Download each metric
    os.makedirs(output_dir, exist_ok=True)
    tasks = []
    for name, img in tqdm(metrics.items(), desc=f"{dataset} → assembling"):
        img_clipped = img.clip(geom)
        desc = f"{dataset}_{name}_{year}"
        out_zip = os.path.join(output_dir, f"{desc}.zip")
        url = get_download_url(img_clipped, bbox, desc)
        tasks.append((url, out_zip))

    with ThreadPoolExecutor() as ex:
        list(tqdm(ex.map(fetch_task, tasks), total=len(tasks), desc=f"Downloading {dataset}"))

    print(f"✅ Completed {dataset} for {year} → {output_dir}")
