#!/usr/bin/env bash
set -euo pipefail

COUNTRY="United Arab Emirates"
START_YEAR=2015
END_YEAR=2015

MASK_TIF="/data/oe23/fert-recon/exports/masks/${COUNTRY}_land_mask.tif"

echo "== Step 1: Export monthly GEE features for ${COUNTRY} (${START_YEAR}-${END_YEAR}) =="
python -m scripts.export_gee_monthly \
  --country "${COUNTRY}" \
  --start_year "${START_YEAR}" \
  --end_year "${END_YEAR}"

echo "== Step 1.5: Export static land mask for ${COUNTRY} (JRC GSW) =="
python -m scripts.export_land_mask \
  --country "${COUNTRY}" \
  --out_tif "${MASK_TIF}" \
  --threshold 50 \
  --scale_m 1000

echo "== Step 2: Build monthly dataframe (land-only) =="

# Wheat
python -m scripts.build_dataframe \
  --start_year "${START_YEAR}" \
  --end_year "${END_YEAR}" \
  --crop wheat \
  --mask_tif "${MASK_TIF}" \
  --out "/data/oe23/fert-recon/datasets/fert_recommendation_monthly_wheat.parquet"

# Maize
python -m scripts.build_dataframe \
  --start_year "${START_YEAR}" \
  --end_year "${END_YEAR}" \
  --crop maize \
  --mask_tif "${MASK_TIF}" \
  --out "/data/oe23/fert-recon/datasets/fert_recommendation_monthly_maize.parquet"
