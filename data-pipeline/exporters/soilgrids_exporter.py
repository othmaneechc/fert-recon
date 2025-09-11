import os
import sys
import ee
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to access shared modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config import TARGET_SCALE_METERS
from config.datasets import DICO
from shared.utils.geometry import country_bbox
from shared.utils.ee_helpers import init_ee, get_download_url, fetch_task


def run_soilgrids(output_dir: str, country: str):
    init_ee()
    # Initialize datasets after EE is ready
    from config.datasets import init_datasets
    init_datasets()
    
    bbox = country_bbox(country)
    layers = DICO['soilgrids']['layers']
    crs = DICO['soilgrids']['crs']
    os.makedirs(output_dir, exist_ok=True)
    tasks = []
    for prop in tqdm(layers, desc='SoilGrids'):
        img = ee.Image(f"projects/soilgrids-isric/{prop}")
        desc = prop
        out_zip = os.path.join(output_dir, f"{desc}.zip")
        url = get_download_url(img, bbox, desc)
        tasks.append((url, out_zip))
    with ThreadPoolExecutor() as ex:
        list(tqdm(ex.map(fetch_task, tasks), total=len(tasks)))