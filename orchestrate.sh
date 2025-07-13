#!/usr/bin/env bash

# ----------------------------------------
# Full pipeline orchestration
# ----------------------------------------

# Path to your coordinate CSV:
COORDS_FILE="coords.csv"

# Years and datasets to loop over
YEARS=($(seq 2000 2015))
REGION_DS=(
#   'modis_ndvi_evi'
#   'modis_lai_fapar'
#   'modis_lst'
#   'modis_et'
#   'chirps'
#   'era5'
#   'terraclimate'
  'srtm_slope'
  'jrc_gsw'
  'modis_mcd12q1'
)

COUNTRY="Morocco"
WORKERS=16

# 1) SoilGrids (static country‐wide)
echo
echo "=== SoilGrids export for $COUNTRY ==="
python3 cli.py soilgrids \
    --country "$COUNTRY" \
    --out "output/soilgrids"

# 2) Region‐based (yearly composites & static)
for YEAR in "${YEARS[@]}"; do
  for DS in "${REGION_DS[@]}"; do
    OUT_DIR="output/region/${DS}/${YEAR}"
    mkdir -p "$OUT_DIR"
    echo
    echo "→ Exporting $DS for $COUNTRY, $YEAR → $OUT_DIR"
    python3 cli.py region "$DS" \
        --year "$YEAR" \
        --country "$COUNTRY" \
        --out "$OUT_DIR"
  done
done

echo
echo "🎉 All exports complete!"
