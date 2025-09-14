#!/usr/bin/env bash
###############################################################################
# pipeline_monthly.sh
#
# Monthly Time Series Pipeline for Transformer-Based Modeling:
#   0. configure → COUNTRY / YEARS / DATASETS
#   1. Export static features (SoilGrids, elevation, etc.) - once per region
#   2. Export monthly time series for dynamic features (NDVI, precipitation, etc.)
#   3. Process all data into transformer-ready time series format
#
# Output: Each pixel has:
#   - Static features: soil properties, elevation, slope, etc.
#   - Monthly sequences: NDVI, precipitation, temperature, etc. for each year
#
# Use cases:
#   - Fertilizer recommendation (predict optimal fertilizer given conditions)
#   - Yield maximization (predict yield given management practices)
###############################################################################
set -euo pipefail

# Change to project root directory
cd "$(dirname "$0")/../.."

# ─────────────────────────── USER SETTINGS ──────────────────────────────
COUNTRY="Morocco"               # passed to cli.py
YEARS=($(seq 2000 2015))        # All years from 2000 to 2015
WORKERS=16                      # if your cli.py supports parallel exports

# Static features (collected once per pixel)
STATIC_FEATURES=(
  soilgrids        # Soil properties: N, sand, silt, clay, pH, etc.
  srtm_slope       # Elevation and slope from SRTM
  jrc_gsw          # Global surface water layers
  modis_mcd12q1    # MODIS land cover classification
)

# Dynamic features (collected monthly for time series)
DYNAMIC_FEATURES=(
  modis_ndvi_evi    # NDVI & EVI monthly means
  modis_lai_fapar   # LAI & fAPAR monthly means  
  modis_lst         # Day & Night LST monthly means
  modis_et          # ET monthly means
  chirps            # CHIRPS rainfall monthly totals
  era5              # ERA5 temperature monthly means
  terraclimate      # TerraClimate variables monthly
)

# ─────────────────────────── STATIC FEATURES EXPORT ────────────────────
echo
echo "===== 1. STATIC FEATURES EXPORT ====="

for FEATURE in "${STATIC_FEATURES[@]}"; do
  echo
  echo "→ Static Feature: $FEATURE ($COUNTRY)"
  
  if [ "$FEATURE" = "soilgrids" ]; then
    python3 data-pipeline/scripts/cli.py soilgrids \
        --country "$COUNTRY" \
        --out "/data/oe23/fert-recon/data/raw/output_monthly/static/$FEATURE"
  else
    # For other static features, use region exporter with a dummy year
    python3 data-pipeline/scripts/cli.py region "$FEATURE" \
        --country "$COUNTRY" \
        --year 2000 \
        --out "/data/oe23/fert-recon/data/raw/output_monthly/static/$FEATURE"
  fi
done

# ─────────────────────────── MONTHLY DYNAMIC FEATURES ─────────────────
echo
echo "===== 2. MONTHLY DYNAMIC FEATURES EXPORT ====="

for YEAR in "${YEARS[@]}"; do
  for FEATURE in "${DYNAMIC_FEATURES[@]}"; do
    
    # Create monthly exports (12 months per year per feature)
    for MONTH in {1..12}; do
      MONTH_PADDED=$(printf "%02d" $MONTH)
      START_DATE="${YEAR}-${MONTH_PADDED}-01"
      
      # Calculate end date (last day of month)
      if [ $MONTH -eq 12 ]; then
        END_DATE="${YEAR}-12-31"
      else
        NEXT_MONTH=$((MONTH + 1))
        NEXT_MONTH_PADDED=$(printf "%02d" $NEXT_MONTH)
        END_DATE=$(date -d "${YEAR}-${NEXT_MONTH_PADDED}-01 - 1 day" +%Y-%m-%d)
      fi
      
      OUT="/data/oe23/fert-recon/data/raw/output_monthly/dynamic/${FEATURE}/${YEAR}/${MONTH_PADDED}"
      mkdir -p "$OUT"
      
      echo
      echo "→ $FEATURE • $COUNTRY • $YEAR-$MONTH_PADDED → $OUT"
      
      python3 data-pipeline/scripts/cli_monthly.py region "$FEATURE" \
          --country "$COUNTRY" \
          --start-date "$START_DATE" \
          --end-date "$END_DATE" \
          --out "$OUT"
    done
  done
done

echo
echo "🎉  All monthly exports finished"

# ─────────────────────────── TIME SERIES PROCESSING ───────────────────
echo
echo "===== 3. Process into Time Series Format ====="
python3 data-pipeline/scripts/create_monthly_time_series.py

echo
echo "🎉  Monthly time series pipeline complete!"
echo "    Static features      : output_monthly/static/"
echo "    Monthly features     : output_monthly/dynamic/"  
echo "    Time series dataset  : data/processed/monthly_time_series_dataset.csv"
echo "    Transformer format   : data/processed/transformer_ready_dataset.npz"