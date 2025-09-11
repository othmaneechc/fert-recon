# Data Directory - Fertility Reconstruction

This directory contains all data files for the fertility reconstruction project.

## Directory Structure

```
data/
├── raw/              # Raw data from various sources
│   ├── output/           # Google Earth Engine extractions
│   └── output_yearly/    # Yearly processing outputs
├── processed/        # Processed and analysis-ready data
│   └── comprehensive_pixels_dataset.csv  # Main modeling dataset
└── README.md        # This file
```

## Data Categories

### Raw Data (`raw/`)
- **Google Earth Engine extractions** (`output/`)
  - Downloaded TIF files by dataset and time period
  - Original spatial resolution and projections
  - Organized by collection and temporal frequency

- **Yearly processing outputs** (`output_yearly/`)
  - Intermediate processing files
  - Yearly aggregations and stacks

### Processed Data (`processed/`)
- **`comprehensive_pixels_dataset.csv`** - Main analysis dataset
  - Format: Each row = one pixel × one year
  - Dimensions: 608,256 rows × 498 columns
  - Size: ~1.5 GB
  - Coverage: 2000-2015 (16 years) × 38,016 pixels
  - Content: Environmental variables for fertility reconstruction

## Data Documentation

### Comprehensive Pixels Dataset
- **Spatial coverage**: 198 × 192 pixels (38,016 total)
- **Temporal coverage**: 2000-2015 (16 years)
- **Variables**: 496 environmental bands including:
  - Climate data (precipitation, temperature, evapotranspiration)
  - Satellite vegetation indices (NDVI, EVI, LST)
  - Surface water characteristics
  - Topographic features
  - Agricultural/cropland information

### Loading the Data
```python
import pandas as pd
df = pd.read_csv('data/processed/comprehensive_pixels_dataset.csv')
```

## Storage Considerations

- **Raw data**: Preserve for reproducibility, may be large
- **Processed data**: Optimized for analysis, regularly backed up
- **Temporary files**: Clean up intermediate processing files periodically

## Data Lineage

1. **Source**: Google Earth Engine collections
2. **Extraction**: Via data pipeline scripts
3. **Processing**: Yearly stacking and aggregation
4. **Integration**: Comprehensive CSV generation
5. **Usage**: Machine learning and analysis

## Backup and Versioning

- Raw data should be preserved as source of truth
- Processed data can be regenerated from raw data
- Consider version control for critical processed datasets
- Document any manual data cleaning or adjustments
