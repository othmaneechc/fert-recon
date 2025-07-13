# Fertilizer Recommendation Pipeline

A structured Python pipeline for downloading and processing multi-source environmental and remote-sensing datasets over arbitrary countries (default: Morocco) at a uniform \~9 km resolution. Outputs per-year summary statistics (mean, min, max, stdDev, percentiles) for temporal collections and static maps for single-image datasets.

## 📂 Directory Structure

```
fert-recon/
├── config.py            # Global constants and EE credentials
├── datasets.py          # Registry of datasets and metadata
├── cli.py               # Command-line interface (soilgrids, region, point)
├── orchestrate.sh       # Bash orchestration (loops years & datasets)
│
├── utils/
│   ├── geometry.py      # Country & point bounding box helpers
│   └── ee_helpers.py    # EarthEngine init and download helpers
│
└── exporters/
    ├── soilgrids_exporter.py  # Bulk download of SoilGrids layers
    ├── region_exporter.py     # Country‑wide yearly composites & static maps
    └── point_exporter.py      # Per-coordinate point exports
```

## ⚙️ Prerequisites

- Python 3.8+
- [Google Earth Engine API](https://developers.google.com/earth-engine)
- `geopandas`, `pandas`, `requests`, `retry`, `tqdm`
- Unix-like shell (for `orchestrate.sh`)

## 📦 Installation

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

## 🔑 Authentication

This pipeline uses a service account with JSON credentials. No interactive auth is required.

```bash
export EE_SERVICE_ACCOUNT="your-service-account@project.iam.gserviceaccount.com"
export EE_KEY_PATH="/path/to/gee_key.json"
```

## 🚀 Usage

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

## 🔧 Configuration

- `config.py` holds default CRS, scale (9 265 m), country name, year range, and credentials.
- `datasets.py` lists all GEE dataset IDs, band selections, native resolutions, and static flags.

## ➕ Adding New Datasets

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

## 📄 License & Contact

This project is licensed under the MIT License. Feel free to open issues or pull requests on GitHub.

Maintainer: [Your Name] ([othmane.echchabi@mail.mcgill.ca](mailto\:othmane.echchabi@mail.mcgill.ca))

---

*Generated based on the Fertilizer Recommendation Pipeline code.*

