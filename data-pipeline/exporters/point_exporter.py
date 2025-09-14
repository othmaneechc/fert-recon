import os, csv, time, logging, sys
from concurrent.futures import ProcessPoolExecutor
import ee
from tqdm import tqdm

# Add parent directory to path to access shared modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config import TARGET_SCALE_METERS
from config.datasets import DICO
from shared.utils.geometry import point_bbox
from shared.utils.ee_helpers import init_ee, get_download_url, fetch_url


def generate_point(coord, dataset, band, sharpened, start_date, end_date, height, width, output_dir):
    lon, lat = coord
    info = DICO[dataset]
    # build geometry
    if info.get('static'):
        img = info['dataset']
    else:
        coll = info['dataset'].filterDate(start_date, end_date)
        img = coll.mean() if dataset in ['harmonize list'] else coll.median()
    bbox = point_bbox(lat, lon, height, TARGET_SCALE_METERS)
    geom = ee.Geometry.Rectangle([[bbox[0], bbox[2]], [bbox[1], bbox[3]]])
    img = img.clip(geom)
    desc = f"{dataset}_{lat:.5f}_{lon:.5f}"
    out_path = os.path.join(output_dir, f"{desc}.tif")
    url = get_download_url(img, geom, desc)
    fetch_url((url, out_path))


def run_point(filepath, dataset, start_date, end_date, height, width, band, sharpened, output_dir, parallel, workers, redownload):
    init_ee()
    
    # Initialize datasets after EE is ready
    from config.datasets import init_datasets
    init_datasets()
    
    logging.basicConfig(level=logging.INFO)
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath) as cf:
        next(cf)
        coords = list(csv.reader(cf, quoting=csv.QUOTE_NONNUMERIC))
    tasks = []
    for c in coords:
        tasks.append((c, dataset, band, sharpened, start_date, end_date, height, width, output_dir))
    if parallel:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(lambda args: generate_point(*args), tasks), total=len(tasks)))
    else:
        for args in tqdm(tasks):
            generate_point(*args)
