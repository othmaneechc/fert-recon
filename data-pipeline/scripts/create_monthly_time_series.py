#!/usr/bin/env python3
"""
Monthly Time Series Extraction and Processing
============================================

Processes monthly exports into time series format suitable for transformer models.

Key features:
- Extracts static features (soil, elevation) once per pixel
- Builds monthly time series for dynamic features
- Creates transformer-ready data structures
- Memory-efficient processing with chunking

Output formats:
1. CSV: pixel_id, static_features, monthly_sequences
2. NPZ: numpy arrays for direct model consumption
"""

import os
import glob
import rasterio
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import gc
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monthly_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MonthlyTimeSeriesBuilder:
    """Builds time series datasets from monthly exports."""
    
    def __init__(self, static_dir="/data/oe23/fert-recon/data/raw/output_monthly/static", 
                 dynamic_dir="/data/oe23/fert-recon/data/raw/output_monthly/dynamic",
                 output_dir="data/processed"):
        self.static_dir = Path(static_dir)
        self.dynamic_dir = Path(dynamic_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset structure
        self.static_features = {}
        self.dynamic_features = {}
        self.pixel_grid = None
        self.years = []
        self.months = list(range(1, 13))
        
    def discover_data_structure(self):
        """Analyze available data to understand structure."""
        logger.info("=== DISCOVERING DATA STRUCTURE ===")
        
        # Discover static features
        if self.static_dir.exists():
            static_folders = [f for f in self.static_dir.iterdir() if f.is_dir()]
            logger.info(f"Found {len(static_folders)} static feature types")
            
            for folder in static_folders:
                feature_name = folder.name
                tif_files = list(folder.glob("*.tif"))
                if tif_files:
                    self.static_features[feature_name] = tif_files[0]  # Take first file
                    logger.info(f"  Static: {feature_name} -> {len(tif_files)} files")
        
        # Discover dynamic features and years
        if self.dynamic_dir.exists():
            dynamic_folders = [f for f in self.dynamic_dir.iterdir() if f.is_dir()]
            logger.info(f"Found {len(dynamic_folders)} dynamic feature types")
            
            for folder in dynamic_folders:
                feature_name = folder.name
                year_folders = [f for f in folder.iterdir() if f.is_dir() and f.name.isdigit()]
                years = sorted([int(f.name) for f in year_folders])
                
                if years:
                    self.dynamic_features[feature_name] = {
                        'years': years,
                        'base_path': folder
                    }
                    logger.info(f"  Dynamic: {feature_name} -> {len(years)} years ({min(years)}-{max(years)})")
                    
                    # Update global years list
                    self.years = sorted(list(set(self.years + years)))
        
        logger.info(f"Overall year range: {min(self.years) if self.years else 'None'} - {max(self.years) if self.years else 'None'}")
        return len(self.static_features) > 0 or len(self.dynamic_features) > 0
    
    def get_reference_grid(self):
        """Get reference pixel grid from the first available raster."""
        logger.info("Establishing reference pixel grid...")
        
        # Try static features first
        for feature_name, tif_path in self.static_features.items():
            try:
                with rasterio.open(tif_path) as src:
                    height, width = src.height, src.width
                    transform = src.transform
                    crs = src.crs
                    
                    self.pixel_grid = {
                        'height': height,
                        'width': width,
                        'total_pixels': height * width,
                        'transform': transform,
                        'crs': crs,
                        'reference_file': str(tif_path)
                    }
                    
                    logger.info(f"Reference grid: {height}x{width} = {height*width:,} pixels")
                    logger.info(f"Reference: {feature_name} ({tif_path.name})")
                    return True
                    
            except Exception as e:
                logger.warning(f"Could not read {tif_path}: {e}")
                continue
        
        # Try dynamic features if static failed
        for feature_name, info in self.dynamic_features.items():
            try:
                # Get first available file
                first_year = info['years'][0]
                first_month_dir = info['base_path'] / str(first_year) / "01"
                if first_month_dir.exists():
                    tif_files = list(first_month_dir.glob("*.tif"))
                    if tif_files:
                        with rasterio.open(tif_files[0]) as src:
                            height, width = src.height, src.width
                            transform = src.transform
                            crs = src.crs
                            
                            self.pixel_grid = {
                                'height': height,
                                'width': width,
                                'total_pixels': height * width,
                                'transform': transform,
                                'crs': crs,
                                'reference_file': str(tif_files[0])
                            }
                            
                            logger.info(f"Reference grid: {height}x{width} = {height*width:,} pixels")
                            logger.info(f"Reference: {feature_name} ({tif_files[0].name})")
                            return True
                            
            except Exception as e:
                logger.warning(f"Could not read dynamic feature {feature_name}: {e}")
                continue
        
        logger.error("Could not establish reference grid from any available data")
        return False
    
    def extract_static_features(self):
        """Extract static features for all pixels."""
        logger.info("=== EXTRACTING STATIC FEATURES ===")
        
        if not self.static_features:
            logger.warning("No static features found")
            return None
        
        total_pixels = self.pixel_grid['total_pixels']
        pixel_ids = np.arange(total_pixels, dtype=np.int32)
        
        static_data = {'pixel_id': pixel_ids}
        
        for feature_name, tif_path in self.static_features.items():
            logger.info(f"Processing static feature: {feature_name}")
            
            try:
                with rasterio.open(tif_path) as src:
                    if src.height != self.pixel_grid['height'] or src.width != self.pixel_grid['width']:
                        logger.warning(f"Size mismatch for {feature_name}, skipping")
                        continue
                    
                    # Read all bands
                    for band_idx in range(src.count):
                        band_data = src.read(band_idx + 1).flatten()
                        
                        # Get band name
                        band_desc = src.descriptions[band_idx] if src.descriptions[band_idx] else f'band_{band_idx+1}'
                        column_name = f"{feature_name}_{band_desc}".replace(' ', '_').replace('.', '_')
                        
                        static_data[column_name] = band_data.astype(np.float32)
                
                logger.info(f"  ✓ {feature_name}: {src.count} bands extracted")
                
            except Exception as e:
                logger.error(f"  ✗ Failed {feature_name}: {e}")
                continue
        
        # Convert to DataFrame
        static_df = pd.DataFrame(static_data)
        logger.info(f"Static features: {static_df.shape[0]:,} pixels, {static_df.shape[1]-1} features")
        
        return static_df
    
    def extract_dynamic_sequences(self, years_to_process=None):
        """Extract monthly time series for dynamic features."""
        logger.info("=== EXTRACTING DYNAMIC TIME SERIES ===")
        
        if not self.dynamic_features:
            logger.warning("No dynamic features found")
            return None
        
        if years_to_process is None:
            years_to_process = self.years
        
        total_pixels = self.pixel_grid['total_pixels']
        
        # Structure: {feature_name: {year: {month: pixel_values}}}
        time_series_data = {}
        
        for feature_name, info in self.dynamic_features.items():
            logger.info(f"Processing dynamic feature: {feature_name}")
            time_series_data[feature_name] = {}
            
            for year in years_to_process:
                if year not in info['years']:
                    logger.warning(f"  Year {year} not available for {feature_name}")
                    continue
                
                time_series_data[feature_name][year] = {}
                year_dir = info['base_path'] / str(year)
                
                for month in self.months:
                    month_str = f"{month:02d}"
                    month_dir = year_dir / month_str
                    
                    if not month_dir.exists():
                        logger.warning(f"  Missing {feature_name} {year}-{month_str}")
                        time_series_data[feature_name][year][month] = np.full(total_pixels, np.nan, dtype=np.float32)
                        continue
                    
                    # Find TIF file in month directory
                    tif_files = list(month_dir.glob("*.tif"))
                    if not tif_files:
                        logger.warning(f"  No TIF files in {month_dir}")
                        time_series_data[feature_name][year][month] = np.full(total_pixels, np.nan, dtype=np.float32)
                        continue
                    
                    try:
                        # Use first TIF file (should be only one)
                        with rasterio.open(tif_files[0]) as src:
                            if src.height != self.pixel_grid['height'] or src.width != self.pixel_grid['width']:
                                logger.warning(f"  Size mismatch for {feature_name} {year}-{month_str}")
                                time_series_data[feature_name][year][month] = np.full(total_pixels, np.nan, dtype=np.float32)
                                continue
                            
                            # For now, take the first band or average all bands
                            if src.count == 1:
                                data = src.read(1).flatten().astype(np.float32)
                            else:
                                # Average all bands (could be configurable)
                                all_bands = src.read().astype(np.float32)
                                data = np.mean(all_bands, axis=0).flatten()
                            
                            time_series_data[feature_name][year][month] = data
                    
                    except Exception as e:
                        logger.warning(f"  Error reading {tif_files[0]}: {e}")
                        time_series_data[feature_name][year][month] = np.full(total_pixels, np.nan, dtype=np.float32)
                
                logger.info(f"  ✓ {feature_name} {year}: 12 months processed")
        
        return time_series_data
    
    def create_transformer_dataset(self, static_df, time_series_data, output_prefix="monthly_time_series"):
        """Create transformer-ready dataset formats."""
        logger.info("=== CREATING TRANSFORMER DATASET ===")
        
        if static_df is None and not time_series_data:
            logger.error("No data to process")
            return None
        
        total_pixels = self.pixel_grid['total_pixels']
        years_to_process = sorted(self.years)
        
        # Create comprehensive CSV format
        csv_data = []
        
        for pixel_id in tqdm(range(total_pixels), desc="Building dataset"):
            pixel_row = {'pixel_id': pixel_id}
            
            # Add static features
            if static_df is not None:
                for col in static_df.columns:
                    if col != 'pixel_id':
                        pixel_row[f"static_{col}"] = static_df.iloc[pixel_id][col]
            
            # Add dynamic features as sequences
            for feature_name, feature_data in time_series_data.items():
                for year in years_to_process:
                    if year in feature_data:
                        for month in self.months:
                            col_name = f"dynamic_{feature_name}_{year}_{month:02d}"
                            if month in feature_data[year]:
                                pixel_row[col_name] = feature_data[year][month][pixel_id]
                            else:
                                pixel_row[col_name] = np.nan
            
            csv_data.append(pixel_row)
            
            # Memory management for large datasets
            if len(csv_data) >= 10000:  # Process in chunks
                df_chunk = pd.DataFrame(csv_data)
                csv_file = self.output_dir / f"{output_prefix}_dataset.csv"
                
                # Write header only for first chunk
                write_header = not csv_file.exists()
                df_chunk.to_csv(csv_file, mode='a', header=write_header, index=False)
                
                csv_data = []
                gc.collect()
        
        # Write remaining data
        if csv_data:
            df_chunk = pd.DataFrame(csv_data)
            csv_file = self.output_dir / f"{output_prefix}_dataset.csv"
            write_header = not csv_file.exists()
            df_chunk.to_csv(csv_file, mode='a', header=write_header, index=False)
        
        logger.info(f"CSV dataset saved: {csv_file}")
        
        # Create NPZ format for models
        self.create_npz_format(static_df, time_series_data, output_prefix)
        
        return csv_file
    
    def create_npz_format(self, static_df, time_series_data, output_prefix):
        """Create NPZ format optimized for transformer models."""
        logger.info("Creating NPZ format for models...")
        
        total_pixels = self.pixel_grid['total_pixels']
        years_to_process = sorted(self.years)
        n_years = len(years_to_process)
        n_months = 12
        
        # Static features array
        if static_df is not None:
            static_features = static_df.drop('pixel_id', axis=1).values.astype(np.float32)
            static_feature_names = [col for col in static_df.columns if col != 'pixel_id']
        else:
            static_features = np.empty((total_pixels, 0), dtype=np.float32)
            static_feature_names = []
        
        # Dynamic features array: [pixels, years, months, features]
        dynamic_feature_names = list(time_series_data.keys())
        n_dynamic_features = len(dynamic_feature_names)
        
        if n_dynamic_features > 0:
            dynamic_features = np.full((total_pixels, n_years, n_months, n_dynamic_features), 
                                     np.nan, dtype=np.float32)
            
            for f_idx, feature_name in enumerate(dynamic_feature_names):
                for y_idx, year in enumerate(years_to_process):
                    if year in time_series_data[feature_name]:
                        for m_idx, month in enumerate(self.months):
                            if month in time_series_data[feature_name][year]:
                                dynamic_features[:, y_idx, m_idx, f_idx] = time_series_data[feature_name][year][month]
        else:
            dynamic_features = np.empty((total_pixels, 0, 0, 0), dtype=np.float32)
        
        # Save NPZ
        npz_file = self.output_dir / f"{output_prefix}_transformer.npz"
        np.savez_compressed(
            npz_file,
            static_features=static_features,
            dynamic_features=dynamic_features,
            pixel_ids=np.arange(total_pixels),
            years=np.array(years_to_process),
            months=np.array(self.months),
            static_feature_names=static_feature_names,
            dynamic_feature_names=dynamic_feature_names,
            grid_info=self.pixel_grid
        )
        
        logger.info(f"NPZ dataset saved: {npz_file}")
        logger.info(f"  Static features: {static_features.shape}")
        logger.info(f"  Dynamic features: {dynamic_features.shape}")
        
        return npz_file

def main():
    """Main execution function."""
    logger.info("=== MONTHLY TIME SERIES BUILDER ===")
    start_time = time.time()
    
    # Initialize builder
    builder = MonthlyTimeSeriesBuilder()
    
    # Discover data structure
    if not builder.discover_data_structure():
        logger.error("No data found to process")
        return
    
    # Establish pixel grid
    if not builder.get_reference_grid():
        logger.error("Could not establish reference grid")
        return
    
    # Extract static features
    static_df = builder.extract_static_features()
    
    # Extract dynamic time series
    time_series_data = builder.extract_dynamic_sequences()
    
    # Create transformer dataset
    if static_df is not None or time_series_data:
        output_file = builder.create_transformer_dataset(static_df, time_series_data)
        
        # Final summary
        total_time = time.time() - start_time
        logger.info(f"\n=== PROCESSING COMPLETE ===")
        logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Output files in: {builder.output_dir}")
        logger.info("🎉 Monthly time series dataset ready for transformer models!")
        
    else:
        logger.error("No data extracted")

if __name__ == "__main__":
    main()