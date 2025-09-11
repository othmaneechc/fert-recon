#!/usr/bin/env python3
"""
Optimized Comprehensive Pixels CSV Generator
===========================================

Efficient processing of all yearly TIF stacks into one CSV with:
- Memory-efficient chunked processing
- Progress tracking
- Error handling
- Output validation

Format: Each row = one pixel in one year
Columns: pixel_id, year, band1, band2, ..., bandN

Author: Fertility Reconstruction Pipeline
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_csv_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_dataset_info(input_dir):
    """Get basic dataset information for planning."""
    logger.info("Analyzing dataset structure...")
    
    tif_files = sorted(glob.glob(os.path.join(input_dir, "yearly_*.tif")))
    if not tif_files:
        raise ValueError(f"No yearly TIF files found in {input_dir}")
    
    # Get reference dimensions from first file
    with rasterio.open(tif_files[0]) as src:
        height, width = src.height, src.width
        num_bands = src.count
        total_pixels = height * width
    
    # Extract years
    years = []
    for tif_file in tif_files:
        filename = os.path.basename(tif_file)
        year_str = filename.replace('yearly_', '').replace('.tif', '')
        try:
            years.append(int(year_str))
        except ValueError:
            continue
    
    years.sort()
    
    info = {
        'files': tif_files,
        'years': years,
        'height': height,
        'width': width,
        'total_pixels': total_pixels,
        'num_bands': num_bands,
        'total_rows': total_pixels * len(years),
        'total_cols': num_bands + 2
    }
    
    # Estimate sizes
    info['memory_gb'] = (info['total_rows'] * info['total_cols'] * 4) / (1024**3)
    info['csv_size_gb'] = (info['total_rows'] * info['total_cols'] * 8) / (1024**3)
    
    logger.info(f"Dataset info: {len(years)} years, {total_pixels:,} pixels/year, {num_bands} bands")
    logger.info(f"Expected output: {info['total_rows']:,} rows, {info['total_cols']} columns")
    logger.info(f"Estimated CSV size: {info['csv_size_gb']:.1f} GB")
    
    return info

def get_standardized_band_names(reference_file):
    """Get clean band names for CSV columns."""
    with rasterio.open(reference_file) as src:
        band_names = []
        for i in range(src.count):
            desc = src.descriptions[i] if src.descriptions[i] else f'band_{i+1}'
            # Clean name for CSV compatibility
            clean_name = (desc.replace('.', '_')
                             .replace('-', '_')
                             .replace(' ', '_')
                             .replace('(', '')
                             .replace(')', '')
                             .replace('/', '_'))
            band_names.append(clean_name)
    return band_names

def process_years_chunk(year_files, band_names, chunk_id):
    """Process a chunk of years efficiently."""
    logger.info(f"Processing chunk {chunk_id} with {len(year_files)} years")
    
    chunk_dataframes = []
    
    for year, tif_file in year_files:
        logger.info(f"  Processing year {year}: {os.path.basename(tif_file)}")
        
        try:
            with rasterio.open(tif_file) as src:
                height, width = src.height, src.width
                num_bands = src.count
                total_pixels = height * width
                
                # Create pixel IDs (row-major order)
                pixel_ids = np.arange(total_pixels, dtype=np.int32)
                
                # Initialize data array for this year
                year_data = np.zeros((total_pixels, min(num_bands, len(band_names)) + 2), dtype=np.float32)
                year_data[:, 0] = pixel_ids  # pixel_id
                year_data[:, 1] = year       # year
                
                # Read bands efficiently
                bands_to_read = min(num_bands, len(band_names))
                for band_idx in range(bands_to_read):
                    band_data = src.read(band_idx + 1)
                    year_data[:, band_idx + 2] = band_data.flatten().astype(np.float32)
                
                # Create DataFrame for this year
                columns = ['pixel_id', 'year'] + band_names[:bands_to_read]
                year_df = pd.DataFrame(year_data, columns=columns)
                year_df['pixel_id'] = year_df['pixel_id'].astype(int)
                year_df['year'] = year_df['year'].astype(int)
                
                chunk_dataframes.append(year_df)
                logger.info(f"    ✓ Year {year}: {year_df.shape[0]:,} rows processed")
                
        except Exception as e:
            logger.error(f"    ✗ Error processing year {year}: {e}")
            continue
    
    if chunk_dataframes:
        # Concatenate all years in chunk
        logger.info(f"Concatenating {len(chunk_dataframes)} years in chunk {chunk_id}")
        chunk_df = pd.concat(chunk_dataframes, ignore_index=True)
        logger.info(f"Chunk {chunk_id} complete: {chunk_df.shape[0]:,} rows")
        return chunk_df
    else:
        logger.warning(f"No data processed in chunk {chunk_id}")
        return None

def create_comprehensive_csv_optimized(input_dir, output_file, years_per_chunk=2):
    """Create comprehensive CSV with optimized memory usage."""
    
    logger.info("=== OPTIMIZED COMPREHENSIVE CSV GENERATION ===")
    start_time = time.time()
    
    # Get dataset information
    info = get_dataset_info(input_dir)
    
    if info['csv_size_gb'] > 20:
        logger.warning(f"⚠️  Very large dataset ({info['csv_size_gb']:.1f} GB)! Consider smaller chunks.")
        years_per_chunk = min(years_per_chunk, 1)
    
    # Get standardized band names
    band_names = get_standardized_band_names(info['files'][0])
    logger.info(f"Using {len(band_names)} standardized band names")
    
    # Create year-to-file mapping
    year_file_map = {}
    for tif_file in info['files']:
        filename = os.path.basename(tif_file)
        year_str = filename.replace('yearly_', '').replace('.tif', '')
        try:
            year = int(year_str)
            year_file_map[year] = tif_file
        except ValueError:
            continue
    
    # Process in chunks
    years = info['years']
    total_chunks = (len(years) + years_per_chunk - 1) // years_per_chunk
    logger.info(f"Processing {len(years)} years in {total_chunks} chunks of {years_per_chunk} years each")
    
    first_chunk = True
    rows_written = 0
    
    for chunk_id in range(total_chunks):
        start_idx = chunk_id * years_per_chunk
        end_idx = min(start_idx + years_per_chunk, len(years))
        chunk_years = years[start_idx:end_idx]
        
        logger.info(f"\n--- CHUNK {chunk_id + 1}/{total_chunks}: Years {chunk_years[0]}-{chunk_years[-1]} ---")
        
        # Prepare year-file pairs for this chunk
        year_files = []
        for year in chunk_years:
            if year in year_file_map:
                year_files.append((year, year_file_map[year]))
        
        # Process chunk
        chunk_df = process_years_chunk(year_files, band_names, chunk_id + 1)
        
        if chunk_df is not None:
            # Write to CSV
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            
            logger.info(f"Writing chunk {chunk_id + 1} to CSV: {chunk_df.shape[0]:,} rows")
            chunk_df.to_csv(output_file, mode=mode, header=header, index=False)
            
            rows_written += chunk_df.shape[0]
            first_chunk = False
            
            # Memory cleanup
            del chunk_df
            gc.collect()
            
            # Progress update
            progress = (chunk_id + 1) / total_chunks * 100
            elapsed = time.time() - start_time
            logger.info(f"Progress: {progress:.1f}% | Rows written: {rows_written:,} | Elapsed: {elapsed:.1f}s")
        
        else:
            logger.warning(f"Chunk {chunk_id + 1} produced no data")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n=== GENERATION COMPLETE ===")
    logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Total rows written: {rows_written:,}")
    logger.info(f"Output file: {output_file}")
    
    if os.path.exists(output_file):
        file_size_mb = os.path.getsize(output_file) / (1024**2)
        logger.info(f"Final file size: {file_size_mb:.1f} MB")
    
    return output_file

def quick_validate_csv(csv_file):
    """Quick validation of the generated CSV."""
    logger.info("=== QUICK VALIDATION ===")
    
    try:
        # Check first few rows
        sample = pd.read_csv(csv_file, nrows=10)
        logger.info(f"✓ CSV readable, {len(sample.columns)} columns")
        logger.info(f"✓ Required columns present: {'pixel_id' in sample.columns and 'year' in sample.columns}")
        
        # Use shell command for row count if available
        try:
            import subprocess
            result = subprocess.run(['wc', '-l', csv_file], capture_output=True, text=True)
            if result.returncode == 0:
                total_rows = int(result.stdout.split()[0]) - 1  # -1 for header
                logger.info(f"✓ Total rows: {total_rows:,}")
        except:
            logger.info("Could not count rows automatically")
        
        logger.info("✓ Validation complete")
        return True
        
    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")
        return False

def main():
    """Main execution function."""
    
    # Configuration
    input_directory = "data/raw/output_yearly/processed/stack_yearly"
    output_filename = "data/processed/comprehensive_pixels_dataset.csv"
    chunk_size = 2  # Process 2 years at a time
    
    # Check inputs
    if not os.path.exists(input_directory):
        logger.error(f"Input directory not found: {input_directory}")
        return
    
    try:
        # Generate the comprehensive CSV
        output_file = create_comprehensive_csv_optimized(
            input_directory, 
            output_filename, 
            years_per_chunk=chunk_size
        )
        
        # Quick validation
        if quick_validate_csv(output_file):
            logger.info("🎉 SUCCESS! Comprehensive CSV generated successfully!")
        else:
            logger.warning("⚠️  Generated CSV may have issues")
        
    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        raise

if __name__ == "__main__":
    main()
