# Fertilizer Recommendation Reconstruction (fert-recon)

## Overview
- Builds monthly, pixel-level datasets linking weather, vegetation, fertilizer inputs, and yields for wheat and maize between 2002 and 2019.
- The data pipeline (under `data-pipeline/`) downloads satellite and reanalysis products from Google Earth Engine, aligns them to the Global Crop Yield 5 arc-minute grid, merges fertilizer rasters, and materializes parquet tables.
- The modeling pipeline (under `modeling-pipeline/`) provides PyTorch sequence models and classical baselines for training yield predictors and simulating fertilizer response.
- Recommendation utilities run coordinate-ascent over fertilizer inputs for the held-out year to produce optimized fertilizer suggestions per pixel.

## Repository Layout
- `data-pipeline/` – Earth Engine exporters, raster alignment utilities, configs, and entry-point scripts.
- `modeling-pipeline/` – training, evaluation, and recommendation code plus experiment configs.
- `datasets_/`, `datasets_chile/`, `datasets_morocco/`, `datasets_spain/` – sample parquet outputs produced by the pipeline.
- `exports_/` – monthly GeoTIFF stacks downloaded from Earth Engine (organized by feature) and masks.
- `models/` – saved experiment outputs (logs, metrics, checkpoints, notebooks).
- `ee_env.yml` – conda environment for the Earth Engine + raster preprocessing stack.

## Setup

### 1. Earth Engine credentials
- Use a service account or user authentication for Google Earth Engine.
- Configure credentials via environment variables: set `GEE_SERVICE_ACCOUNT` and `GEE_PRIVATE_KEY` (path to the key file or the JSON string), or set `GOOGLE_APPLICATION_CREDENTIALS` to a service account key path. Avoid storing secrets in the repository.
- The exporters fall back to interactive auth (`ee.Initialize()`), but service accounts are recommended for unattended runs.

### 2. Conda environment for the data pipeline
```bash
conda env create -f ee_env.yml
conda activate ee
```
- The environment installs Earth Engine, rasterio, pyarrow, xarray, and helper libraries used across `data-pipeline/scripts/`.

### 3. Environment for modeling
- Create or reuse a Python ≥3.10 environment with PyTorch, pandas, numpy, scikit-learn, pyarrow, matplotlib, and tqdm. XGBoost is optional (`pip install xgboost`) for tree baselines.
- Ensure GPU drivers are configured if you plan to train the transformer/LSTM models on CUDA.

## Data Pipeline

1. **Configure paths and AOI**  
   Adjust `data-pipeline/config/config.py` to point to local fertilizer rasters (`data/Cropland_Maps`), yield NetCDF files (`data/GlobalCropYield5min`), export directories, and the default country (Morocco by default).

2. **Export monthly covariates from Earth Engine**  
   ```bash
   cd data-pipeline
   python -m scripts.export_gee_monthly --country <COUNTRY> --start_year 2002 --end_year 2019
   ```
   - The datasets pulled are defined in `config/datasets.py` (CHIRPS precipitation, ERA5-Land weather, MODIS vegetation & LST, derived VPD).
   - Outputs land under `exports_/monthly/<feature>/<YYYY>/<YYYYMM>.tif`.

3. **Optional: export a static land mask**  
   ```bash
   python -m scripts.export_land_mask --country <COUNTRY> --out_tif exports_/masks/<COUNTRY>_land_mask.tif --threshold 50
   ```

4. **Build the aligned monthly parquet table**  
   ```bash
   python -m scripts.build_dataframe \
     --start_year 2002 \
     --end_year 2019 \
     --crop wheat \
     --mask_tif exports_/masks/<COUNTRY>_land_mask.tif \
     --out datasets_/fert_recommendation_monthly_wheat.parquet
   ```
   - Repeat for maize (switch `--crop maize`).
   - The script reprojects/averages monthly rasters to the yield grid, joins Cropland_Maps fertilizer bands, and writes one row per pixel-month with fertilizer, weather, vegetation, and yield targets.

5. **End-to-end helper**  
   `run_end_to_end.sh` demonstrates the full pipeline (feature export → mask → parquet) for a single country/year range; customize the variables at the top before running.

## Modeling Pipeline

- Configurations live in `modeling-pipeline/configs/` and support hierarchical includes plus `${...}` interpolation (`src/utils/io.py`).
  - `dataset_common.yaml` centralizes dataset parameters (paths, splits, feature lists, normalization).
  - `featuresets.yaml` defines named subsets for ablations.
  - Model-specific YAMLs (e.g., `transformer_base.yaml`, `lstm_base.yaml`, `elasticnet_log.yaml`) extend the common settings.

### Training
```bash
cd modeling-pipeline
python -m src.train --config configs/transformer_base.yaml
```
- The trainer instantiates the dataset (`MonthlySequenceDataset`), applies cleaning, normalization, and temporal splits, then trains the requested model (transformer, LSTM, TCN, linear/elastic net, tree-based, or XGBoost if available).
- Metrics, plots, and checkpoints land under `../models/exp_logs/<timestamp>_...`. See the generated log file for the exact run directory.

### Evaluation and analysis
- `src.eval` can reload a checkpoint and compute metrics on validation/test years or alternative feature subsets.
- `modeling-pipeline/notebooks/data_diagnostics.ipynb` offers exploratory plots on the parquet tables.
- Saved models and notebooks in `models/` (e.g., `model_analysis.ipynb`) show usage patterns.

## Fertilizer Recommendation Workflow
```bash
python -m src.recommend \
  --config configs/transformer_base.yaml \
  --ckpt ../models/<run>/best_wheat_temporal.pt \
  --out_csv ../models/<run>/recommendations_wheat.csv
```
- Loads the trained transformer, filters the dataset to the held-out test year, and performs coordinate ascent over fertilizer features (`fert_*`) bounded by the 1st–99th percentile of historical inputs.
- The output CSV captures `pixel_id`, `year`, base prediction, optimized prediction, and recommended fertilizer levels per pixel.

## Data & Outputs
- `datasets_/`, `datasets_chile/`, `datasets_morocco/`, `datasets_spain/` store sample parquet files for wheat and maize.
- `exports_/` holds intermediate GeoTIFFs and masks from Earth Engine runs.
- `models/` organizes experiment artifacts; subdirectories correspond to different feature sets or algorithms.

## Tips & Troubleshooting
- Large Earth Engine exports may take several minutes per month; reruns skip files unless `OVERWRITE_GEE_EXPORTS` is set to `True` in the config.
- If `ee.Initialize()` fails, double-check service account permissions and that `gee_key.json` is readable by the active environment.
- Verify that `data/Cropland_Maps` and `data/GlobalCropYield5min` contain the expected TIFF/NetCDF assets before running `build_dataframe.py`.
- Use the provided notebooks to inspect data coverage and model outputs before launching large training jobs.
