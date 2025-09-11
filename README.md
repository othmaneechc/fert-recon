# Fertility Reconstruction Pipeline

A comprehensive pipeline for downloading, processing, and modeling multi-source environmental and remote-sensing datasets for fertility reconstruction analysis. The pipeline consists of two main components: a data pipeline for data extraction and processing, and a modeling pipeline for machine learning and analysis.

## Directory Structure

```
fert-recon/
├── data-pipeline/           # Data extraction and processing
│   ├── scripts/                # Core processing scripts
│   ├── config/                # Configuration files
│   ├── exporters/             # Data exporters
│   ├── ee_env.yml             # Environment file
│   └── pipeline.sh            # Main pipeline script
│
├── modeling-pipeline/       # Machine learning and analysis
│   ├── notebooks/             # Jupyter notebooks
│   ├── scripts/               # Production modeling scripts
│   ├── models/                # Trained model artifacts
│   └── experiments/           # Experiment tracking
│
├── shared/                  # Shared utilities
│   └── utils/                 # Common helper functions
│
├── data/                    # Data storage
│   ├── raw/                   # Original downloaded data
│   └── processed/             # Analysis-ready datasets
│
├── docs/                    # Documentation
└── README.md               # This file
```

## Prerequisites

- Python 3.8+
- [Google Earth Engine API](https://developers.google.com/earth-engine)
- `geopandas`, `pandas`, `requests`, `retry`, `tqdm`
- Unix-like shell (for `orchestrate.sh`)

## Installation

1. Clone this repository:
   ```bash
   git clone git@github.com:<your-username>/fert-recon.git
   cd fert-recon
   ```
2. Create and activate a Conda or virtualenv environment:
   To install the rest of the required packages, the user can create a conda environment similar to the one we use. Our environment file can be found in `ee_env.yml`. Installing a conda environment using a yml file is done through: conda env create -f YML_FILE_NAME 

   You might need to run the following command:
   ```bash
   conda install -c conda-forge google-cloud-sdk
   ```

4. Place your GEE JSON key at `gee_key.json` or set `EE_KEY_PATH` env var.
5. Ensure your coordinate CSV (`coords.csv`) is in the project root.

## Authentication

This pipeline uses a service account with JSON credentials. No interactive auth is required.

```bash
export EE_SERVICE_ACCOUNT="your-service-account@project.iam.gserviceaccount.com"
export EE_KEY_PATH="/path/to/gee_key.json"
```

## Usage

### 1. SoilGrids (static soil property layers)

```bash
bash orchestrate.sh
```

Downloads all SoilGrids layers at 5′ (\~9 km) resolution:

```
output/soilgrids/
  nitrogen_mean.zip
  sand_mean.zip
  …
```

### 2. Region-based Exports (yearly composites & static)

By default, loops **2000–2015** over datasets configured in `orchestrate.sh`:

- Temporal: MODIS (`MOD13Q1`, `MCD15A3H`, `MOD11A1`, `MOD16A2`), CHIRPS, ERA5-Land, TerraClimate
- Static: SRTM slope, JRC surface water, MODIS land-cover (`MCD12Q1`)

```bash
# Single-run for one year/dataset:
python3 cli.py region modis_ndvi_evi --year 2022 --country Morocco --out output/region/modis_ndvi_evi/2022
```

Outputs:

```
output/region/modis_ndvi_evi/2022/
  modis_ndvi_evi_mean_2022.zip
  modis_ndvi_evi_min_2022.zip
  …
```

### 3. Point-based Exports (per-coordinate)

*(Optional)* Export a square window around each lat/lon in `coords.csv`:

```bash
python3 cli.py point sentinel --coords coords.csv --year 2022 --parallel --workers 16 --out output/point/sentinel
```

## Configuration

- `config.py` holds default CRS, scale (9 265 m), country name, year range, and credentials.
- `datasets.py` lists all GEE dataset IDs, band selections, native resolutions, and static flags.

## Adding New Datasets

1. In `datasets.py`, add a new key:
   ```python
   DICO['your_ds'] = {
       'dataset': ee.ImageCollection('GEE/ID'),
       'bands': {...},
       'resolution': 1000,
       'min': 0, 'max': 100,
       # 'static': True  # for non-temporal single-image datasets
   }
   ```
2. If temporal, it will follow the composite pipeline (mean, min, max, etc.).
3. If static, it will be clipped & downloaded once per year.

## License & Contact

This project is licensed under the MIT License. Feel free to open issues or pull requests on GitHub.

Maintainer: Othmane Echchabi ([othmane.echchabi@mail.mcgill.ca](mailto\:othmane.echchabi@mail.mcgill.ca))

---

*Generated based on the Fertilizer Recommendation Pipeline code.*

