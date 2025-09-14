#!/usr/bin/env python3
"""
Monthly Region Exporter for Time Series Data
============================================

Exports monthly aggregated raster data for transformer-based modeling.
This replaces the yearly statistics approach with monthly time series.

Key features:
- Monthly mean/median/sum aggregations
- Proper handling of different temporal frequencies
- Optimized for time series construction
- Memory-efficient processing
"""

import os
import sys
import ee
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to access shared modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config import TARGET_SCALE_METERS
from shared.utils.geometry import country_bbox
from shared.utils.ee_helpers import init_ee, get_download_url, fetch_url

# Datasets that benefit from different aggregation methods
PRECIPITATION_DS = ['chirps']  # Use sum for precipitation
TEMPERATURE_DS = ['era5', 'modis_lst']  # Use mean for temperature
DEFAULT_AGGREGATION = 'mean'  # Default for NDVI, EVI, LAI, etc.

def get_aggregation_method(dataset, user_aggregation):
    """Determine the best aggregation method for each dataset."""
    if user_aggregation != 'mean':
        return user_aggregation  # User override
    
    if dataset in PRECIPITATION_DS:
        return 'sum'  # Total monthly precipitation
    elif dataset in TEMPERATURE_DS:
        return 'mean'  # Average monthly temperature
    else:
        return 'mean'  # Default for vegetation indices, etc.

def run_region_monthly(dataset: str, start_date: str, end_date: str, 
                      output_dir: str, country: str, aggregation: str = 'mean'):
    """
    Export monthly aggregated raster data for a specific dataset and date range.
    
    Args:
        dataset: Dataset name from DICO
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory
        country: Country name
        aggregation: Aggregation method ('mean', 'median', 'sum', 'max', 'min')
    """
    
    # Initialize Earth Engine
    init_ee()
    
    # Initialize datasets after EE is ready
    from config import datasets
    datasets.init_datasets()

    # Get country bounding box
    bbox = country_bbox(country)
    geom = ee.Geometry.Rectangle(bbox)
    
    # Get actual aggregation method for this dataset
    actual_aggregation = get_aggregation_method(dataset, aggregation)
    
    print(f"Processing {dataset} for {start_date} to {end_date}")
    print(f"Using {actual_aggregation} aggregation")

    # Handle static datasets
    if datasets.DICO[dataset].get('static', False):
        print(f"Note: {dataset} is static, ignoring date range")
        ds_obj = datasets.DICO[dataset]['dataset']
        
        if hasattr(ds_obj, 'filterDate'):
            # It's an ImageCollection, get the first image
            coll = ds_obj.filterBounds(geom)
            if coll.size().getInfo() == 0:
                print(f"No images for {dataset}, skipping.")
                return
            img = coll.first()
        else:
            # It's already an Image
            img = ds_obj
    else:
        # Dynamic dataset - filter by date and aggregate
        ds_obj = datasets.DICO[dataset]['dataset']
        
        # Filter collection by date and bounds
        coll = ds_obj.filterDate(start_date, end_date).filterBounds(geom)
        
        if coll.size().getInfo() == 0:
            print(f"No images for {dataset} in {start_date} to {end_date}, skipping.")
            return
        
        # Apply temporal aggregation
        if actual_aggregation == 'mean':
            img = coll.mean()
        elif actual_aggregation == 'median':
            img = coll.median()
        elif actual_aggregation == 'sum':
            img = coll.sum()
        elif actual_aggregation == 'max':
            img = coll.max()
        elif actual_aggregation == 'min':
            img = coll.min()
        else:
            raise ValueError(f"Unknown aggregation method: {actual_aggregation}")
    
    # Clip the image to the country boundary
    img_clipped = img.clip(geom)
    
    # Create output filename
    os.makedirs(output_dir, exist_ok=True)
    start_month = start_date[:7].replace('-', '_')  # YYYY_MM format
    filename = f"{dataset}_{start_month}_{actual_aggregation}"
    out_zip = os.path.join(output_dir, f"{filename}.zip")
    
    # Get download URL and download immediately (fast approach)
    print(f"Getting download URL for: {filename}")
    url = get_download_url(img_clipped, bbox, filename)
    fetch_url((url, out_zip))
    
    print(f"✓ Export completed: {filename} → {out_zip}")

def run_batch_monthly(datasets: list, year: int, country: str, 
                     output_base: str, aggregation: str = 'mean'):
    """
    Run monthly exports for multiple datasets and all months in a year.
    
    Args:
        datasets: List of dataset names
        year: Year to process
        country: Country name
        output_base: Base output directory
        aggregation: Default aggregation method
    """
    
    print(f"=== Batch Monthly Export: {year} ===")
    print(f"Datasets: {datasets}")
    print(f"Country: {country}")
    
    for month in range(1, 13):
        month_str = f"{month:02d}"
        start_date = f"{year}-{month_str}-01"
        
        # Calculate last day of month
        if month == 12:
            end_date = f"{year}-12-31"
        else:
            next_month = month + 1
            if next_month == 2 and year % 4 == 0:  # Leap year February
                end_date = f"{year}-02-29"
            elif next_month == 2:
                end_date = f"{year}-02-28"
            elif next_month in [4, 6, 9, 11]:  # 30-day months
                end_date = f"{year}-{next_month:02d}-30"
            else:  # 31-day months
                end_date = f"{year}-{next_month:02d}-31"
            
            # Adjust to last day of current month
            import datetime
            last_day = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
            end_date = last_day.strftime("%Y-%m-%d")
        
        print(f"\n--- Month {month_str}/{year}: {start_date} to {end_date} ---")
        
        for dataset in datasets:
            output_dir = os.path.join(output_base, dataset, str(year), month_str)
            
            try:
                run_region_monthly(dataset, start_date, end_date, 
                                 output_dir, country, aggregation)
            except Exception as e:
                print(f"✗ Failed {dataset} {year}-{month_str}: {e}")
                continue

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Monthly region exporter')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('--start-date', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--country', default='Morocco', help='Country name')
    parser.add_argument('--out', default='output_monthly', help='Output directory')
    parser.add_argument('--aggregation', default='mean', help='Aggregation method')
    
    args = parser.parse_args()
    
    run_region_monthly(args.dataset, args.start_date, args.end_date,
                      args.out, args.country, args.aggregation)