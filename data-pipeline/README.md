# Data Pipeline - Fertility Reconstruction

This pipeline handles data extraction, processing, and preparation for the fertility reconstruction project.

## Overview

The data pipeline consists of several stages:
1. **Data Extraction** from Google Earth Engine
2. **Processing** into yearly stacks
3. **CSV Generation** for modeling

## Directory Structure

```
data-pipeline/
├── scripts/           # Main processing scripts
├── config/            # Configuration files
├── exporters/         # Data exporters for different sources
├── ee_env.yml        # Conda environment file
└── pipeline.sh       # Main pipeline execution script
```

## Key Scripts

### Configuration
- `config/datasets.py` - Dataset definitions and metadata
- `config/config.py` - Main configuration settings

### Core Processing
- `scripts/cli.py` - Command line interface
- `scripts/gee_extract_stack.py` - Google Earth Engine data extraction
- `scripts/extract_all.py` - Stack processing and extraction
- `scripts/create_comprehensive_pixels_csv_optimized.py` - CSV generation

### Utilities
- `scripts/dataframe.py` - DataFrame utilities
- `exporters/` - Specialized exporters for different data sources

## Usage

### 1. Environment Setup
```bash
conda env create -f ee_env.yml
conda activate ee
```

### 2. Run Full Pipeline
```bash
./pipeline.sh
```

### 3. Individual Steps

#### Extract Data from Google Earth Engine
```bash
python scripts/cli.py extract --region your_region
```

#### Process into Yearly Stacks
```bash
python scripts/extract_all.py
```

#### Generate Comprehensive CSV
```bash
python scripts/create_comprehensive_pixels_csv_optimized.py
```

## Output

The pipeline generates:
- **Raw TIF files** in `../data/raw/output/`
- **Yearly stacks** in `../data/raw/output/processed/stack_yearly/`
- **Final CSV dataset** in `../data/processed/comprehensive_pixels_dataset.csv`

## Configuration

### Dataset Configuration
Edit `config/datasets.py` to modify:
- Available datasets
- Temporal frequencies (daily, monthly, yearly)
- Processing parameters

### Regional Configuration
Edit `config/config.py` to modify:
- Study region boundaries
- Output paths
- Processing parameters

## Dependencies

See `ee_env.yml` for complete environment requirements:
- Google Earth Engine API
- Rasterio for geospatial processing
- Pandas for data manipulation
- NumPy for numerical operations

## Notes

- Requires Google Earth Engine authentication
- Large datasets may require significant processing time
- Monitor disk space for large regional extractions
